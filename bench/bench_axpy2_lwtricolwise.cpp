#include "coeff_matrix_2d.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>

using namespace dso;

namespace {

using Mat = CoeffMatrix2D<MatrixStorageType::LwTriangularColWise>;

/* Fill a lower-triangular matrix with deterministic values.
 * This keeps setup reproducible and avoids RNG overhead/noise.
 */
static void fill_matrix(Mat &M, double alpha, double beta) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j <= i; ++j) {
      M(i, j) = alpha * static_cast<double>(i + 1) +
                beta * static_cast<double>(j + 1);
    }
  }
}

/* Benchmark the explicit two-source in-place kernel:
 *   A.axpy2_inplace(s1, B1, s2, B2)
 *
 * We reset A outside the timed region so that the benchmark measures
 * only the target operation.
 */
static void BM_Axpy2Direct(benchmark::State &state) {
  const int n = static_cast<int>(state.range(0));
  const double s1 = -1.75;
  const double s2 = 0.625;

  Mat A0(n);
  Mat A(n);
  Mat B1(n);
  Mat B2(n);

  /* One-time initialization before timing starts. */
  fill_matrix(A0, 0.25, -0.50);
  fill_matrix(B1, 1.50, 0.125);
  fill_matrix(B2, -0.75, 0.375);

  A = A0;

  const std::size_t nelem = static_cast<std::size_t>(A.num_elements());

#ifdef DSO_SIMD
  state.SetLabel("simd");
#else
  state.SetLabel("scalar");
#endif

  for (auto _ : state) {
    /* Reset A outside timed region. */
    state.PauseTiming();
    A = A0;
    state.ResumeTiming();

    /* Timed operation. */
    A.axpy2_inplace(s1, B1, s2, B2);

    /* Prevent dead-code elimination. */
    benchmark::DoNotOptimize(A);
    benchmark::ClobberMemory();
  }

  /* Traffic model:
   * - read A
   * - read B1
   * - read B2
   * - write A
   *
   * That is 4 * nelem doubles touched per iteration.
   */
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(nelem));
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(
                                                   4 * nelem * sizeof(double)));
}

} // namespace

/* Dimensions:
 * 20, 40, 60, ..., 300
 */
BENCHMARK(BM_Axpy2Direct)->DenseRange(20, 300, 20);

BENCHMARK_MAIN();