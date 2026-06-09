#include "eigen3/Eigen/Eigen"
#include "gravity.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kRe = 6378137.0;
constexpr double kGM = 3.986004418e14;

/*
 * Touch a buffer larger than typical LLC size so each sh2gradient call starts
 * from a "cold-ish" cache state. This work is intentionally kept *inside* the
 * timed region because the user requested that calls should be performed after
 * cache clearing to mimic real conditions.
 */
inline void flush_cache(std::vector<std::uint64_t> &buffer) {
  std::uint64_t sum = 0;
  constexpr std::size_t stride = 8; /* 8 * 8 bytes = 64-byte cache line */
  for (std::size_t i = 0; i < buffer.size(); i += stride) {
    buffer[i] += 0x9e3779b97f4a7c15ULL;
    sum ^= buffer[i];
  }
  benchmark::DoNotOptimize(sum);
  benchmark::ClobberMemory();
}

/*
 * Deterministically populate a test field. Assumes StokesCoeffs exposes
 * Cnm()/Snm() matrix accessors and a constructor accepting (degree, order).
 * If your exact constructor differs, only this one line may need adjusting.
 */
dso::StokesCoeffs make_test_field(const int degree) {
  dso::StokesCoeffs cs(degree, degree);

  auto &C = cs.Cnm();
  auto &S = cs.Snm();

  std::mt19937_64 rng(123456789ULL + static_cast<std::uint64_t>(degree));
  std::uniform_real_distribution<double> dist(-1e-6, 1e-6);

  /* Keep a dominant zonal term and small higher-degree content. */
  C(0, 0) = 1.0;
  for (int n = 1; n <= degree; ++n) {
    for (int m = 0; m <= n; ++m) {
      C(n, m) = dist(rng);
      if (m > 0) {
        S(n, m) = dist(rng);
      }
    }
    /* Optional light decay with degree to look more realistic. */
    const double scale = 1.0 / static_cast<double>((n + 1) * (n + 1));
    for (int m = 0; m <= n; ++m) {
      C(n, m) *= scale;
      if (m > 0) {
        S(n, m) *= scale;
      }
    }
  }

  return cs;
}

std::vector<Eigen::Vector3d> make_points(const int calls) {
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(static_cast<std::size_t>(calls));

  for (int i = 0; i < calls; ++i) {
    /* Exterior points with mild variation. */
    const double x = 1.08 * kRe + 1000.0 * static_cast<double>(i % 17);
    const double y = 0.13 * kRe + 750.0 * static_cast<double>(i % 19);
    const double z = 0.21 * kRe + 500.0 * static_cast<double>(i % 23);
    pts.emplace_back(x, y, z);
  }

  return pts;
}

void BenchmarkArgs(benchmark::internal::Benchmark *b) {
  constexpr int sizes[] = {4, 10, 50, 100, 150, 180, 190};
  constexpr int calls[] = {5, 10, 20, 50, 100, 150, 200, 300, 400, 500};

  for (const int n : sizes) {
    for (const int c : calls) {
      b->Args({n, c});
    }
  }
}

void BM_sh2gradient(benchmark::State &state) {
  const int degree = static_cast<int>(state.range(0));
  const int calls = static_cast<int>(state.range(1));

  dso::StokesCoeffs cs = make_test_field(degree);
  std::vector<Eigen::Vector3d> points = make_points(calls);

  /* Scratch space reused across calls. sh2gradient needs degree+2 inclusive,
   * so dimension count degree+3 is the safe pool size.
   */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> W(degree + 3);
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> M(degree + 3);

  /* 64 MiB cache trash buffer. */
  std::vector<std::uint64_t> cache_buffer(
      (64u * 1024u * 1024u) / sizeof(std::uint64_t), 1ULL);

  Eigen::Vector3d acc;
  Eigen::Matrix<double, 3, 3> grad;
  double sink = 0.0;

  for (auto _ : state) {
    for (int i = 0; i < calls; ++i) {
      // flush_cache(cache_buffer);
      const int status =
          dso::sh2gradient(cs, points[static_cast<std::size_t>(i)], acc, grad,
                           degree, degree, kRe, kGM, &W, &M);
      if (status != 0) {
        throw std::runtime_error(
            "dso::sh2gradient returned non-zero status in benchmark");
      }
      sink += acc.x() + grad(0, 0);
      benchmark::DoNotOptimize(acc);
      benchmark::DoNotOptimize(grad);
      benchmark::ClobberMemory();
    }
  }

  benchmark::DoNotOptimize(sink);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(calls));
  state.counters["degree"] = static_cast<double>(degree);
  state.counters["calls_per_iter"] = static_cast<double>(calls);
  state.counters["ns_per_call"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(calls),
      benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

BENCHMARK(BM_sh2gradient)->Apply(BenchmarkArgs)->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
