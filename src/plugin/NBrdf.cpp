/*
 * NBrdf.cpp -- Neural BRDF (NBRDF) plugin for Pixar's RenderMan (RIS).
 *
 * v3 (diagnostic) changes, on top of v2:
 *   - New optional int parameter "samplingMode": 0 = mixture (default,
 *     same as v2), 1 = cosine-hemisphere ONLY, 2 = GGX-around-mirror
 *     ONLY. Lets one compiled DLL be pointed at the furnace test
 *     (furnace_check.py) three times -- once per technique -- to
 *     localize an energy-conservation mismatch to one specific pdf
 *     formula (cosine or GGX) versus the 0.5/0.5 mixture combination
 *     itself, without a rebuild per variant. This is also exactly the
 *     "three plugin variants" ablation performance-evaluation-protocol.md
 *     Track B asks for -- one build now serves both purposes.
 *   - Nothing about the sampling MATH changed from v2; mode 0 is
 *     bit-for-bit the v2 behavior.
 *
 * v2 changes:
 *   - HID_DIM increased from 21 to 64 (4,803 parameters)
 *   - Sampling is now a 50/50 MIXTURE of cosine-hemisphere sampling
 *     (broad -- explores the whole hemisphere, including the
 *     grazing-angle brightening this material actually has) and GGX
 *     half-vector sampling around the mirror direction (narrow --
 *     nails the true, very tight specular spike). A single narrow GGX
 *     lobe alone was tried first and made things WORSE: the trained
 *     network shows this material is darkest near the classic mirror
 *     direction and brightest at grazing angles, so concentrating all
 *     samples at the mirror direction just samples the dark region.
 *     The mixture pdf (MixturePDF) is reported consistently from
 *     GenerateSample, EvaluateSample, and EvaluateSamplesAtIndex so
 *     MIS with light sampling stays correct.
 *
 * Evaluates a small trained neural network in place of an analytic
 * reflectance model. The network (6 -> 64 -> 64 -> 3, 4803 parameters)
 * is trained offline in PyTorch on a MERL BRDF material and exported
 * as a flat float32 blob, which this plugin loads at instance time
 * via the "weightsPath" parameter.
 *
 * Method follows Sztrajman et al., "Neural BRDF Representation and
 * Importance Sampling" (CGF 2021): Rusinkiewicz half/difference
 * parametrization as network input, log-space-trained reflectance
 * output. Reflectance-only (a BRDF, not a full BSDF) -- MERL has no
 * transmission data, so GetEvaluateDomain reports k_RixBXReflect.
 *
 * ---------------------------------------------------------------
 * LOBE ACTIVATION -- the fix for the long-standing crash.
 *
 * The correct pattern is AddActiveLobe(), which registers the lobe
 * and returns a valid, writable per-point array. Applied in all
 * three shading methods below; any one left unfixed reintroduces
 * the corruption intermittently.
 * ---------------------------------------------------------------
 *
 * Build (MSVC x64 Native Tools Command Prompt):
 *   cl /std:c++17 /EHsc /D_USE_MATH_DEFINES ^
 *      /FI"compat\gcc_builtins.h" /I"compat" /I"%RMANTREE%\include" ^
 *      /c NBrdf.cpp
 *   link /DLL /OUT:NBrdf.dll NBrdf.obj "%RMANTREE%\lib\libpxrcore.lib"
 */

#include <RixBxdf.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// =================================================================
// Network definition and inference
// =================================================================

constexpr int IN_DIM = 6, HID_DIM = 64, OUT_DIM = 3;

// Expected total weight count: 6*64 + 64 + 64*64 + 64 + 64*3 + 3 = 4803
constexpr int TOTAL_WEIGHTS = IN_DIM*HID_DIM + HID_DIM +
                              HID_DIM*HID_DIM + HID_DIM +
                              HID_DIM*OUT_DIM + OUT_DIM;

struct NBrdfWeights {
    float W1[IN_DIM][HID_DIM],   b1[HID_DIM];
    float W2[HID_DIM][HID_DIM],  b2[HID_DIM];
    float W3[HID_DIM][OUT_DIM],  b3[OUT_DIM];
    bool  loaded = false;

    static double SumArray(const float* p, size_t n) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += p[i];
        return s;
    }

    // Sum of every weight and bias. Two different trained materials give
    // different values, so the render log can show which weights a given
    // material really loaded and shaded with.
    double Checksum() const {
        return SumArray(&W1[0][0], IN_DIM * HID_DIM) + SumArray(b1, HID_DIM) +
               SumArray(&W2[0][0], HID_DIM * HID_DIM) + SumArray(b2, HID_DIM) +
               SumArray(&W3[0][0], HID_DIM * OUT_DIM) + SumArray(b3, OUT_DIM);
    }

    bool LoadFromFile(const char* path) {
        if (!path || path[0] == '\0') {
            fprintf(stderr, "NBrdf: empty weightsPath\n");
            return false;
        }
        FILE* f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "NBrdf: failed to open weights file '%s'\n", path);
            return false;
        }
        bool ok = true;
        ok &= fread(W1, sizeof(float), IN_DIM * HID_DIM,  f) == (size_t)(IN_DIM * HID_DIM);
        ok &= fread(b1, sizeof(float), HID_DIM,           f) == (size_t)HID_DIM;
        ok &= fread(W2, sizeof(float), HID_DIM * HID_DIM, f) == (size_t)(HID_DIM * HID_DIM);
        ok &= fread(b2, sizeof(float), HID_DIM,           f) == (size_t)HID_DIM;
        ok &= fread(W3, sizeof(float), HID_DIM * OUT_DIM, f) == (size_t)(HID_DIM * OUT_DIM);
        ok &= fread(b3, sizeof(float), OUT_DIM,           f) == (size_t)OUT_DIM;
        // A correct file ends exactly here. Leftover bytes mean it came from
        // a different (e.g. wider) network, which would otherwise load
        // silently as garbage.
        bool tooLong = ok && (fgetc(f) != EOF);
        fclose(f);
        if (!ok) {
            fprintf(stderr, "NBrdf: weights file '%s' has wrong size "
                            "(expected %d float32 = %d bytes)\n",
                            path, TOTAL_WEIGHTS, (int)(TOTAL_WEIGHTS * sizeof(float)));
        } else if (tooLong) {
            fprintf(stderr, "NBrdf: weights file '%s' is LONGER than expected "
                            "(%d float32 = %d bytes) -- wrong network size?\n",
                            path, TOTAL_WEIGHTS, (int)(TOTAL_WEIGHTS * sizeof(float)));
            ok = false;
        } else {
            fprintf(stderr, "NBrdf: loaded '%s' (%d floats, checksum %.6f)\n",
                    path, TOTAL_WEIGHTS, Checksum());
        }
        loaded = ok;
        return ok;
    }
};

inline float Relu(float x) { return x > 0.0f ? x : 0.0f; }

void EvalNetwork(const NBrdfWeights& w, const float in[IN_DIM], float out[OUT_DIM]) {
    float h1[HID_DIM];
    for (int j = 0; j < HID_DIM; ++j) {
        float acc = w.b1[j];
        for (int i = 0; i < IN_DIM; ++i) acc += in[i] * w.W1[i][j];
        h1[j] = Relu(acc);
    }
    float h2[HID_DIM];
    for (int j = 0; j < HID_DIM; ++j) {
        float acc = w.b2[j];
        for (int i = 0; i < HID_DIM; ++i) acc += h1[i] * w.W2[i][j];
        h2[j] = Relu(acc);
    }
    for (int j = 0; j < OUT_DIM; ++j) {
        float acc = w.b3[j];
        for (int i = 0; i < HID_DIM; ++i) acc += h2[i] * w.W3[i][j];
        out[j] = std::fmax(0.0f, std::exp(acc) - 1e-3f);
    }
}

// =================================================================
// Vector math and the Rusinkiewicz half/difference parametrization
// =================================================================

struct Vec3 { float x, y, z; };

inline Vec3  Add(Vec3 a, Vec3 b)   { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3  Scale(Vec3 v, float s) { return {v.x*s, v.y*s, v.z*s}; }
inline Vec3  Sub(Vec3 a, Vec3 b)   { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float Dot(Vec3 a, Vec3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3  Cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline Vec3 Normalize(Vec3 v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return len < 1e-8f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{v.x/len, v.y/len, v.z/len};
}
inline float Length(Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
inline Vec3 RotateZ(Vec3 v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return {c*v.x - s*v.y, s*v.x + c*v.y, v.z};
}
inline Vec3 RotateY(Vec3 v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return {c*v.x + s*v.z, v.y, -s*v.x + c*v.z};
}

inline Vec3 ToLocalFrame(Vec3 worldDir, Vec3 T, Vec3 B, Vec3 N) {
    return {Dot(worldDir, T), Dot(worldDir, B), Dot(worldDir, N)};
}

inline Vec3 ToWorldFrame(Vec3 localDir, Vec3 T, Vec3 B, Vec3 N) {
    return {T.x*localDir.x + B.x*localDir.y + N.x*localDir.z,
            T.y*localDir.x + B.y*localDir.y + N.y*localDir.z,
            T.z*localDir.x + B.z*localDir.y + N.z*localDir.z};
}

inline void BuildFrame(Vec3 Nin, Vec3 Tin, Vec3& N, Vec3& T, Vec3& B) {
    N = Normalize(Nin);
    B = Normalize(Cross(N, Tin));
    T = Cross(B, N);
}

void StandardToHalfDiff(Vec3 wi, Vec3 wo, float& theta_h, float& theta_d, float& phi_d) {
    Vec3 h = Normalize(Add(wi, wo));
    theta_h = std::acos(std::fmax(-1.0f, std::fmin(1.0f, h.z)));
    float phi_h = std::atan2(h.y, h.x);

    Vec3 d = RotateY(RotateZ(wi, -phi_h), -theta_h);
    theta_d = std::acos(std::fmax(-1.0f, std::fmin(1.0f, d.z)));
    phi_d = std::atan2(d.y, d.x);
    if (phi_d < 0.0f) phi_d += float(M_PI);
}

void EvalReflectance(const NBrdfWeights& w, Vec3 wiLocal, Vec3 woLocal, float rgb[3]) {
    float theta_h, theta_d, phi_d;
    StandardToHalfDiff(wiLocal, woLocal, theta_h, theta_d, phi_d);

    Vec3 h = {std::sin(theta_h), 0.0f, std::cos(theta_h)};
    Vec3 d = {std::sin(theta_d) * std::cos(phi_d),
              std::sin(theta_d) * std::sin(phi_d),
              std::cos(theta_d)};

    float in[IN_DIM] = {h.x, h.y, h.z, d.x, d.y, d.z};
    EvalNetwork(w, in, rgb);
}

// =================================================================
// GGX importance sampling (Walter et al. 2007)
// =================================================================

// GGX roughness parameter. For alum-bronze (a fairly specular
// metallic), 0.05 is a reasonable starting point. Increase for
// rougher materials (e.g. 0.2-0.5 for satin/matte).
constexpr float GGX_ALPHA = 0.05f;

// GGX normal distribution function
inline float GGX_D(float cosTheta, float alpha) {
    float a2 = alpha * alpha;
    float cos2 = cosTheta * cosTheta;
    float denom = cos2 * (a2 - 1.0f) + 1.0f;
    return a2 / (float(M_PI) * denom * denom);
}

// Sample a half-vector from GGX distribution in local frame.
// Returns the sampled half-vector (local space, z = up).
inline Vec3 SampleGGX(float u1, float u2, float alpha) {
    float a2 = alpha * alpha;
    float cosTheta = std::sqrt((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1));
    float sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * float(M_PI) * u2;
    return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
}

// PDF of sampling direction wi given view direction wo, using GGX
// half-vector sampling. pdf = D(h) * cos(theta_h) / (4 * |wo . h|)
inline float GGX_PDF(Vec3 wiLocal, Vec3 woLocal, float alpha) {
    Vec3 h = Normalize(Add(wiLocal, woLocal));
    float cosTheta_h = std::fmax(0.0f, h.z);
    float D = GGX_D(cosTheta_h, alpha);
    // v3 fix (2026-10-14): the old guard (`if (woDotH < 1e-8f) return 0`)
    // only caught the exact-zero case. Anything just above that -- which
    // happens routinely at near-grazing wo, since h stays close to N for
    // small alpha and woDotH ~= cos(theta_o) there -- still divides by a
    // near-zero number and returns a huge-but-finite pdf. Fed into a
    // one-sample MIS estimator, that's a textbook firefly: suggestive
    // evidence being the max|diff| outliers (6.4-6.7, vs. an expected
    // ceiling of ~0.19) seen in BOTH the furnace test and the A2 distant-
    // light test. Clamping the denominator to a floor instead of only
    // guarding the zero case is the standard production mitigation.
    float woDotH = std::fmax(std::fabs(Dot(woLocal, h)), 1e-3f);
    return D * cosTheta_h / (4.0f * woDotH);
}

// Reflect direction d around half-vector h.
inline Vec3 Reflect(Vec3 d, Vec3 h) {
    float doth = Dot(d, h);
    return Sub(Scale(h, 2.0f * doth), d);
}

// Cosine-weighted hemisphere pdf, used as the "broad" half of the mixture
// below. On its own this is what v1 used; it's a poor match for the tiny
// specular spike, but -- as the training-data slice plot for this
// material shows -- it's actually a GOOD match for where most of this
// material's energy lives (a broad lobe that brightens toward grazing
// angles, not a simple peak at the mirror direction).
inline float CosinePDF(Vec3 wiLocal) {
    return std::fmax(0.0f, wiLocal.z) / float(M_PI);
}

inline Vec3 SampleCosineHemisphere(float u1, float u2) {
    float r   = std::sqrt(u1);
    float phi = 2.0f * float(M_PI) * u2;
    return {r * std::cos(phi), r * std::sin(phi),
            std::sqrt(std::fmax(0.0f, 1.0f - u1))};
}

// Mixture pdf: half cosine-hemisphere (broad -- captures grazing sheen),
// half GGX-around-mirror (narrow -- captures the true specular spike).
// A single narrow GGX lobe alone under-samples everywhere except right at
// the mirror direction, which for this material is actually one of the
// DARKEST regions -- hence the earlier all-black render. Mixing in the
// cosine term fixes that without losing the spike.
inline float MixturePDF(Vec3 wiLocal, Vec3 woLocal, float alpha) {
    return 0.5f * CosinePDF(wiLocal) + 0.5f * GGX_PDF(wiLocal, woLocal, alpha);
}

// =================================================================
// Diagnostic sampling-mode switch (v3). Mode 0 reproduces v2 exactly;
// modes 1/2 isolate one technique for the furnace-test ablation
// described in the file header. Keep this logic trivial and close to
// the mixture code above so the three modes stay easy to audit against
// each other -- the point is to localize a bug, not to add a new one.
// =================================================================
enum SamplingMode { kMixture = 0, kCosineOnly = 1, kGgxOnly = 2 };

// Sample wi given wo (local frame) under the requested mode. `technique`
// is the same DrawSample1D() draw GenerateSample already pulls per point;
// passed in rather than redrawn so RNG usage stays identical across modes
// for mode 0 (no behavior change) and across comparable draws for 1/2.
inline Vec3 SampleMode(int mode, float technique, RtFloat2 xi, Vec3 woLocal, float alpha) {
    bool useCosine;
    switch (mode) {
        case kCosineOnly: useCosine = true;  break;
        case kGgxOnly:     useCosine = false; break;
        default:           useCosine = technique < 0.5f; break;  // kMixture
    }
    if (useCosine) {
        return SampleCosineHemisphere(xi.x, xi.y);
    }
    Vec3 hLocal = SampleGGX(xi.x, xi.y, alpha);
    return Reflect(woLocal, hLocal);
}

// Pdf of wi given wo (local frame) under the requested mode. Mode 0 is
// MixturePDF unchanged; modes 1/2 report the SINGLE technique's own pdf
// (not the 0.5-weighted mixture pdf), since in single-technique mode
// that is the actual sampling density -- using the mixture formula here
// would itself be a bug, not a diagnostic.
inline float PdfMode(int mode, Vec3 wiLocal, Vec3 woLocal, float alpha) {
    switch (mode) {
        case kCosineOnly: return CosinePDF(wiLocal);
        case kGgxOnly:     return GGX_PDF(wiLocal, woLocal, alpha);
        default:           return MixturePDF(wiLocal, woLocal, alpha);  // kMixture
    }
}

// =================================================================
// Plugin constants
// =================================================================

const RixBXLobeTraits kOurLobe(false, true, true, false, 0x1, 0x0, 0x0);

inline RixBXLobeSampled OurLobeSampled() {
    RixBXLobeSampled s;
    s.Set(false, false, true, false, false, 0, 0);
    return s;
}

RtUString const kWeightsPathParam("weightsPath");
RtUString const kSamplingModeParam("samplingMode");

}  // namespace

// =================================================================
// Per-shading-context Bxdf
// =================================================================

class NBrdfBxdf : public RixBxdf {
public:
    NBrdfBxdf(RixShadingContext const* sCtx, RixBxdfFactory* factory,
              NBrdfWeights const* w, int mode)
        : RixBxdf(sCtx, factory), weights(w), samplingMode(mode) {}

    RixBXEvaluateDomain GetEvaluateDomain() override { return k_RixBXReflect; }

    void GetAggregateLobeTraits(RixBXLobeTraits* t) override { *t = kOurLobe; }

    RixSCDetail GetProperty(BxdfProperty, void const**) const override {
        return k_RixSCInvalidDetail;
    }

    bool EmitLocal(RtColorRGB*) override { return false; }

    // -------------------------------------------------------------
    // BRDF-sampling side of MIS: generate an outgoing direction and
    // report its pdf, plus the reflectance for that direction.
    //
    // v2: 50/50 mixture of cosine-hemisphere sampling and GGX
    // half-vector sampling. The GGX half samples a half-vector and
    // reflects the view direction around it to get the incident
    // direction (narrow, around the mirror direction); the cosine half
    // covers the broad grazing-angle response. See MixturePDF above.
    // -------------------------------------------------------------
    void GenerateSample(RixBXTransportTrait, RixBXLobeTraits const*, RixRNG* rng,
                        RixBXLobeSampled* lobesSampled, RtVector3* directions,
                        RixBXLobeWeights& weightsOut, float* forwardPdfs, float* reversePdfs,
                        RtColorRGB* compTrans = NULL) override {
        RixShadingContext const* sCtx = GetShadingCtx();
        RtFloat3 const* Nn; sCtx->GetBuiltinVar(RixShadingContext::k_Nn, &Nn);
        RtFloat3 const* Vn; sCtx->GetBuiltinVar(RixShadingContext::k_Vn, &Vn);
        RtFloat3 const* Tn; sCtx->GetBuiltinVar(RixShadingContext::k_Tn, &Tn);

        RixBXLobeSampled sampled = OurLobeSampled();
        RtColorRGB* lobe = weightsOut.AddActiveLobe(sampled);

        for (int i = 0; i < sCtx->numPts; ++i) {
            Vec3 N, T, B;
            BuildFrame({Nn[i].x, Nn[i].y, Nn[i].z}, {Tn[i].x, Tn[i].y, Tn[i].z}, N, T, B);
            Vec3 V = {Vn[i].x, Vn[i].y, Vn[i].z};
            Vec3 woLocal = ToLocalFrame(V, T, B, N);

            // Mixture sampling: 50% broad cosine-hemisphere, 50% narrow
            // GGX-around-mirror. Picking only the GGX lobe concentrates
            // every sample right at the mirror direction -- which, per
            // the training-data plot, is actually one of the DARKEST
            // regions for this material. The cosine half covers the
            // broader grazing-angle brightening; the GGX half still
            // nails the narrow true specular spike when present.
            // (v3: samplingMode selects which technique(s) are live --
            // see SampleMode/PdfMode above. samplingMode==0 reproduces
            // this exactly, same RNG draws, same branch condition.)
            float technique = rng->DrawSample1D(i);
            RtFloat2 xi = rng->DrawSample2D(i);
            Vec3 wiLocal = SampleMode(samplingMode, technique, xi, woLocal, GGX_ALPHA);

            // Check that the sampled direction is in the upper hemisphere
            if (wiLocal.z <= 0.0f) {
                // Below hemisphere -- kill this sample
                lobe[i] = RtColorRGB(0.0f, 0.0f, 0.0f);
                forwardPdfs[i] = 0.0f;
                if (reversePdfs) reversePdfs[i] = 0.0f;
                lobesSampled[i] = RixBXLobeSampled();  // default ctor => invalid/zeroed
                directions[i] = RtVector3(0.0f, 0.0f, 1.0f);
                continue;
            }

            Vec3 wiWorld = Normalize(ToWorldFrame(wiLocal, T, B, N));
            directions[i] = RtVector3(wiWorld.x, wiWorld.y, wiWorld.z);

            float rgb[3] = {0.5f, 0.5f, 0.5f};
            if (weights->loaded) EvalReflectance(*weights, wiLocal, woLocal, rgb);

            lobe[i] = RtColorRGB(rgb[0], rgb[1], rgb[2]);

            // Mixture pdf -- matches whichever technique actually could
            // have produced this direction, weighted by how likely each
            // technique was to pick it (standard one-sample MIS).
            forwardPdfs[i] = PdfMode(samplingMode, wiLocal, woLocal, GGX_ALPHA);
            if (reversePdfs) reversePdfs[i] = PdfMode(samplingMode, woLocal, wiLocal, GGX_ALPHA);
            lobesSampled[i] = sampled;
        }
    }

    // -------------------------------------------------------------
    // Light-sampling side of MIS: the integrator supplies the incident
    // direction; report reflectance and pdf for it.
    // -------------------------------------------------------------
    void EvaluateSample(RixBXTransportTrait, RixBXLobeTraits const*, RixRNG*,
                        RixBXLobeTraits* lobesEvaluated, RtVector3 const* directions,
                        RixBXLobeWeights& weightsOut, float* forwardPdfs,
                        float* reversePdfs) override {
        RixShadingContext const* sCtx = GetShadingCtx();
        RtFloat3 const* Nn; sCtx->GetBuiltinVar(RixShadingContext::k_Nn, &Nn);
        RtFloat3 const* Vn; sCtx->GetBuiltinVar(RixShadingContext::k_Vn, &Vn);
        RtFloat3 const* Tn; sCtx->GetBuiltinVar(RixShadingContext::k_Tn, &Tn);

        RixBXLobeSampled sampled = OurLobeSampled();
        RtColorRGB* lobe = weightsOut.AddActiveLobe(sampled);

        for (int i = 0; i < sCtx->numPts; ++i) {
            Vec3 N, T, B;
            BuildFrame({Nn[i].x, Nn[i].y, Nn[i].z}, {Tn[i].x, Tn[i].y, Tn[i].z}, N, T, B);
            Vec3 V = {Vn[i].x, Vn[i].y, Vn[i].z};

            Vec3 Lworld  = {directions[i].x, directions[i].y, directions[i].z};
            Vec3 wiLocal = ToLocalFrame(Lworld, T, B, N);
            Vec3 woLocal = ToLocalFrame(V, T, B, N);

            float rgb[3] = {0.5f, 0.5f, 0.5f};
            if (weights->loaded) EvalReflectance(*weights, wiLocal, woLocal, rgb);

            lobe[i] = RtColorRGB(rgb[0], rgb[1], rgb[2]);

            // Report the mixture pdf for MIS weighting (must match
            // GenerateSample's actual sampling density).
            forwardPdfs[i] = PdfMode(samplingMode, wiLocal, woLocal, GGX_ALPHA);
            if (reversePdfs) reversePdfs[i] = PdfMode(samplingMode, woLocal, wiLocal, GGX_ALPHA);
            if (lobesEvaluated) lobesEvaluated[i] = kOurLobe;
        }
    }

    // -------------------------------------------------------------
    // Same as EvaluateSample, but numSamples directions at one fixed
    // shading point `index` rather than one direction across points.
    // -------------------------------------------------------------
    void EvaluateSamplesAtIndex(RixBXTransportTrait, RixBXLobeTraits const&, RixRNG*, int index,
                                int numSamples, RixBXLobeTraits* lobesEvaluated,
                                RtVector3 const* directions, RixBXLobeWeights& weightsOut,
                                float* forwardPdfs, float* reversePdfs) override {
        RixShadingContext const* sCtx = GetShadingCtx();
        RtFloat3 const* Nn; sCtx->GetBuiltinVar(RixShadingContext::k_Nn, &Nn);
        RtFloat3 const* Vn; sCtx->GetBuiltinVar(RixShadingContext::k_Vn, &Vn);
        RtFloat3 const* Tn; sCtx->GetBuiltinVar(RixShadingContext::k_Tn, &Tn);

        Vec3 N, T, B;
        BuildFrame({Nn[index].x, Nn[index].y, Nn[index].z},
                   {Tn[index].x, Tn[index].y, Tn[index].z}, N, T, B);
        Vec3 V = {Vn[index].x, Vn[index].y, Vn[index].z};
        Vec3 woLocal = ToLocalFrame(V, T, B, N);

        RixBXLobeSampled sampled = OurLobeSampled();
        RtColorRGB* lobe = weightsOut.AddActiveLobe(sampled);

        for (int s = 0; s < numSamples; ++s) {
            Vec3 Lworld  = {directions[s].x, directions[s].y, directions[s].z};
            Vec3 wiLocal = ToLocalFrame(Lworld, T, B, N);

            float rgb[3] = {0.5f, 0.5f, 0.5f};
            if (weights->loaded) EvalReflectance(*weights, wiLocal, woLocal, rgb);

            lobe[s] = RtColorRGB(rgb[0], rgb[1], rgb[2]);
            forwardPdfs[s] = PdfMode(samplingMode, wiLocal, woLocal, GGX_ALPHA);
            if (reversePdfs) reversePdfs[s] = PdfMode(samplingMode, woLocal, wiLocal, GGX_ALPHA);
            if (lobesEvaluated) lobesEvaluated[s] = kOurLobe;
        }
    }

private:
    NBrdfWeights const* weights;
    int samplingMode;  // 0=mixture (v2 behavior), 1=cosine-only, 2=GGX-only
};

// =================================================================
// Factory
// =================================================================

class NBrdfFactory : public RixBxdfFactory {
public:
    int  Init(RixContext&, RtUString const) override { return 0; }
    void Finalize(RixContext&) override {}

    RixSCParamInfo const* GetParamTable() override {
        static RixSCParamInfo table[] = {
            RixSCParamInfo(RtUString("weightsPath"), k_RixSCString, k_RixSCInput),
            // v3 diagnostic param: 0=mixture (default/v2 behavior),
            // 1=cosine-only, 2=GGX-only. See SamplingMode enum above.
            RixSCParamInfo(RtUString("samplingMode"), k_RixSCInteger, k_RixSCInput),
            RixSCParamInfo()
        };
        return table;
    }

    void Synchronize(RixContext&, RixSCSyncMsg, RixParameterList const*) override {}

    void CreateInstanceData(RixContext&, RtUString const handle,
                            RixParameterList const* parameterList,
                            InstanceData*) override {
        if (!parameterList) return;
        int paramId;
        if (parameterList->GetParamId(kWeightsPathParam, &paramId) == 0) {
            RtUString path;
            parameterList->EvalParam(paramId, 0, &path);
            // RixBxdf.h: one factory per bound Bxdf request, so each
            // material handle should report its own factory pointer here.
            fprintf(stderr, "NBrdf: factory %p, handle '%s', weightsPath '%s'\n",
                    (void*)this, handle.CStr(), path.CStr());
            weights.LoadFromFile(path.CStr());
        }
        // v3 diagnostic param. Optional -- missing/unset stays at the
        // default of 0 (mixture, i.e. identical to v2) set in the member
        // declaration below, so existing RIBs that don't set this at all
        // are unaffected.
        if (parameterList->GetParamId(kSamplingModeParam, &paramId) == 0) {
            int mode = 0;
            parameterList->EvalParam(paramId, 0, &mode);
            if (mode < 0 || mode > 2) {
                fprintf(stderr, "NBrdf: samplingMode %d out of range [0,2], using 0 (mixture)\n", mode);
                mode = 0;
            }
            samplingMode = mode;
            fprintf(stderr, "NBrdf: factory %p, handle '%s', samplingMode=%d "
                            "(0=mixture 1=cosine-only 2=GGX-only)\n",
                    (void*)this, handle.CStr(), samplingMode);
        }
    }

    int  GetInstanceHints(void*) const override { return k_TriviallyOpaque; }
    void RegisterTemporalVolumeParams(void*, std::vector<int>&) const override {}

    RixBxdf* BeginScatter(RixShadingContext const* sCtx, RixBXLobeTraits const&,
                          RixSCShadingMode, void*, void*) override {
        // Once per factory: report the weights this material shades with.
        if (!loggedFirstScatter.exchange(true))
            fprintf(stderr, "NBrdf: factory %p shading with checksum %.6f (loaded=%d)\n",
                    (void*)this, weights.Checksum(), (int)weights.loaded);

        RixShadingContext::Allocator alloc(sCtx);
        NBrdfBxdf* bxdf = alloc.Allocate<NBrdfBxdf>(1);
        return new (bxdf) NBrdfBxdf(sCtx, this, &weights, samplingMode);
    }
    void EndScatter(RixBxdf*) override {}

    RixOpacity* BeginOpacity(RixShadingContext const*, RixSCShadingMode, void*, void*) override { return NULL; }
    void EndOpacity(RixOpacity*) override {}
    RixVolumeIntegrator* BeginInterior(RixShadingContext const*, RixSCShadingMode, void*, void*) override { return NULL; }
    void EndInterior(RixVolumeIntegrator*) override {}
    RixVolumeIntegrator* BeginSubsurface(RixShadingContext const*, RixSCShadingMode, void*, void*) override { return NULL; }
    void EndSubsurface(RixVolumeIntegrator*) override {}
    float GetIndexOfRefraction(void*) const override { return 1.0f; }

    NBrdfWeights weights;
    int samplingMode = 0;  // 0=mixture (default/v2 behavior), set in CreateInstanceData
    std::atomic<bool> loggedFirstScatter{false};
};

// =================================================================
// Plugin entry points
// =================================================================

RIX_BXDFPLUGINCREATE
{
    return new NBrdfFactory();
}

RIX_BXDFPLUGINDESTROY
{
    delete static_cast<NBrdfFactory*>(bxdf);
}
