#include "coeff_matrix_2d.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>

using namespace dso;

namespace {

using Mat = CoeffMatrix2D<MatrixStorageType::LwTriangularColWise>;

/* Fill a lower-triangular matrix with deterministic values.
 * Avoid random-number generators here because benchmarking should measure
 * the target operation only, not setup overhead or RNG noise.
 */
static void fill_matrix(Mat &M, double alpha, double beta) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j <= i; ++j) {
      M(i, j) = alpha * static_cast<double>(i + 1) +
                beta * static_cast<double>(j + 1);
    }
  }
}

/* Benchmark the public syntax:
 *   A += s * B
 *
 * This is the operation users will actually write. In the SIMD build,
 * the specialized overload / kernel should be selected automatically.
 * In the scalar build, the normal fallback path will run.
 */
static void BM_AxpyOpPlusEqScaled(benchmark::State &state) {
  const int n = static_cast<int>(state.range(0));
  const double s = -1.75;

  Mat A0(n);
  Mat A(n);
  Mat B(n);

  /* Initialize the matrices once before timing starts. */
  fill_matrix(A0, 0.25, -0.50);
  fill_matrix(B, 1.50, 0.125);
  A = A0;

  const std::size_t nelem = static_cast<std::size_t>(A.num_elements());

#ifdef DSO_SIMD
  state.SetLabel("simd");
#else
  state.SetLabel("scalar");
#endif

  for (auto _ : state) {
    /* Reset A outside the timed region so the benchmark measures only the
     * target operation and not the setup/reset work.
     */
    state.PauseTiming();
    A = A0;
    state.ResumeTiming();

    /* Timed operation. */
    A += s * B;

    /* Prevent the compiler from proving the result is unused. */
    benchmark::DoNotOptimize(A);
    benchmark::ClobberMemory();
  }

  /* AXPY-like traffic model:
   * - read A
   * - read B
   * - write A
   * That is 3 * nelem doubles per iteration.
   */
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(nelem));
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(
                                                   3 * nelem * sizeof(double)));
}

/* Optional second benchmark: call the dedicated kernel-facing member directly.
 * This removes any small expression-template overhead from the measurement and
 * isolates the actual in-place axpy path.
 *
 * Keep this benchmark only if we have:
 *   void axpy_inplace(double s, const CoeffMatrix2D& rhs) noexcept;
 */
static void BM_AxpyDirect(benchmark::State &state) {
  const int n = static_cast<int>(state.range(0));
  const double s = -1.75;

  Mat A0(n);
  Mat A(n);
  Mat B(n);

  fill_matrix(A0, 0.25, -0.50);
  fill_matrix(B, 1.50, 0.125);
  A = A0;

  const std::size_t nelem = static_cast<std::size_t>(A.num_elements());

#ifdef DSO_SIMD
  state.SetLabel("simd");
#else
  state.SetLabel("scalar");
#endif

  for (auto _ : state) {
    state.PauseTiming();
    A = A0;
    state.ResumeTiming();

    A.axpy_inplace(s, B);

    benchmark::DoNotOptimize(A);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(nelem));
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(
                                                   3 * nelem * sizeof(double)));
}

} // namespace

/* Dimensions requested:
 * 20, 40, 60, ..., 300
 */
BENCHMARK(BM_AxpyOpPlusEqScaled)->DenseRange(20, 300, 20);
BENCHMARK(BM_AxpyDirect)->DenseRange(20, 300, 20);

BENCHMARK_MAIN();