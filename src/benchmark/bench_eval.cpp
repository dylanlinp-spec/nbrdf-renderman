/*
 * bench_eval.cpp -- standalone microbenchmark for the NBRDF plugin's
 * per-shading-point cost, WITHOUT RenderMan. It reuses the exact same
 * network layout, StandardToHalfDiff(), EvalNetwork() and EvalReflectance()
 * as NBrdf_v2.cpp, so the nanoseconds reported here are the per-evaluation
 * cost the renderer pays inside GenerateSample / EvaluateSample.
 *
 * It times three things separately so the write-up can say where the time
 * goes:
 *   1. EvalReflectance  -- angle conversion + full network (what the plugin calls)
 *   2. EvalNetwork only -- network forward pass on a pre-built input
 *   3. StandardToHalfDiff only -- the Rusinkiewicz angle conversion + trig
 *
 * Build (release -- this MUST be optimized or the numbers are meaningless):
 *   MSVC:  cl /std:c++17 /O2 /EHsc /D_USE_MATH_DEFINES bench_eval.cpp
 *   g++:   g++ -std=c++17 -O2 bench_eval.cpp -o bench_eval
 *
 * Run:
 *   bench_eval nbrdf_weights.bin [n_evals]
 *   (n_evals default 10,000,000)
 *
 * Report the median of >=5 runs. Pin to one core if you can
 * (Windows: start /affinity 1 ; Linux: taskset -c 0).
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Network layout + inference -- COPIED VERBATIM from NBrdf_v2.cpp.
// Keep in sync if the architecture changes.
// ============================================================
constexpr int IN_DIM = 6, HID_DIM = 64, OUT_DIM = 3;
constexpr int TOTAL_WEIGHTS = IN_DIM*HID_DIM + HID_DIM +
                              HID_DIM*HID_DIM + HID_DIM +
                              HID_DIM*OUT_DIM + OUT_DIM;

struct NBrdfWeights {
    float W1[IN_DIM][HID_DIM],   b1[HID_DIM];
    float W2[HID_DIM][HID_DIM],  b2[HID_DIM];
    float W3[HID_DIM][OUT_DIM],  b3[OUT_DIM];
    bool  loaded = false;

    bool LoadFromFile(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "bench: cannot open '%s'\n", path); return false; }
        bool ok = true;
        ok &= fread(W1, sizeof(float), IN_DIM * HID_DIM,  f) == (size_t)(IN_DIM * HID_DIM);
        ok &= fread(b1, sizeof(float), HID_DIM,           f) == (size_t)HID_DIM;
        ok &= fread(W2, sizeof(float), HID_DIM * HID_DIM, f) == (size_t)(HID_DIM * HID_DIM);
        ok &= fread(b2, sizeof(float), HID_DIM,           f) == (size_t)HID_DIM;
        ok &= fread(W3, sizeof(float), HID_DIM * OUT_DIM, f) == (size_t)(HID_DIM * OUT_DIM);
        ok &= fread(b3, sizeof(float), OUT_DIM,           f) == (size_t)OUT_DIM;
        fclose(f);
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

struct Vec3 { float x, y, z; };
inline Vec3  Add(Vec3 a, Vec3 b)    { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3  Normalize(Vec3 v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return len < 1e-8f ? Vec3{0,0,1} : Vec3{v.x/len, v.y/len, v.z/len};
}
inline Vec3 RotateZ(Vec3 v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return {c*v.x - s*v.y, s*v.x + c*v.y, v.z};
}
inline Vec3 RotateY(Vec3 v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return {c*v.x + s*v.z, v.y, -s*v.x + c*v.z};
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

// ============================================================
// Benchmark driver
// ============================================================
using Clock = std::chrono::steady_clock;
static double ns_per(Clock::time_point a, Clock::time_point b, long n) {
    return std::chrono::duration<double, std::nano>(b - a).count() / double(n);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin [n_evals]\n", argv[0]); return 2; }
    long N = (argc >= 3) ? std::atol(argv[2]) : 10000000L;

    NBrdfWeights w;
    if (!w.LoadFromFile(argv[1])) return 1;

    // Precompute N random upper-hemisphere (wi, wo) pairs so random-number
    // generation is OUT of the timed region.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<Vec3> WI(N), WO(N);
    for (long k = 0; k < N; ++k) {
        for (Vec3* v : {&WI[k], &WO[k]}) {
            float z = U(rng);                        // cos(theta) in (0,1] => upper hemisphere
            float r = std::sqrt(std::fmax(0.0f, 1.0f - z*z));
            float p = 2.0f * float(M_PI) * U(rng);
            *v = {r*std::cos(p), r*std::sin(p), z};
        }
    }

    volatile float sink = 0.0f;  // defeat dead-code elimination

    // ---- 1. Full EvalReflectance (what the plugin calls per sample) ----
    {
        auto t0 = Clock::now();
        for (long k = 0; k < N; ++k) {
            float rgb[3];
            EvalReflectance(w, WI[k], WO[k], rgb);
            sink += rgb[0] + rgb[1] + rgb[2];
        }
        auto t1 = Clock::now();
        printf("EvalReflectance    : %8.2f ns/eval  (%ld evals)\n", ns_per(t0, t1, N), N);
    }

    // ---- 2. Angle conversion only ----
    {
        auto t0 = Clock::now();
        for (long k = 0; k < N; ++k) {
            float th, td, pd;
            StandardToHalfDiff(WI[k], WO[k], th, td, pd);
            sink += th + td + pd;
        }
        auto t1 = Clock::now();
        printf("StandardToHalfDiff : %8.2f ns/eval\n", ns_per(t0, t1, N));
    }

    // ---- 3. Network only (input prebuilt once; isolates MLP arithmetic) ----
    {
        float in[IN_DIM] = {0.1f, 0.0f, 0.99f, 0.2f, 0.1f, 0.97f};
        auto t0 = Clock::now();
        for (long k = 0; k < N; ++k) {
            float rgb[3];
            in[0] = WI[k].x * 0.01f;  // vary input cheaply so it can't be hoisted
            EvalNetwork(w, in, rgb);
            sink += rgb[0] + rgb[1] + rgb[2];
        }
        auto t1 = Clock::now();
        printf("EvalNetwork only   : %8.2f ns/eval\n", ns_per(t0, t1, N));
    }

    fprintf(stderr, "(sink=%g)\n", (float)sink);
    return 0;
}
