#include "coeff_matrix_2d.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace dso;

#ifdef DSO_SIMD
namespace {
bool almost_equal(double a, double b, double eps = 1e-13) {
  return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}
} // namespace
#endif

int main() {
#ifndef DSO_SIMD
  std::puts("DSO_SIMD not enabled; skipping SIMD-specific test.");
  return 0;
#else
  using Mat = CoeffMatrix2D<MatrixStorageType::LwTriangularColWise>;

  /* Pick a size whose number of stored elements is not a multiple of 4.
   * For n=23, num_elements = 23*24/2 = 276.
   * 276 = 8*34 + 4, so we exercise both the 8-wide and 4-wide AVX loops.
   *
   * If we want to force scalar tail too, use n=22:
   * 22*23/2 = 253 = 8*31 + 5, so we get vector work plus leftover scalar work.
   */
  constexpr int N = 22;
  constexpr double s = -1.75;

  Mat A(N);
  Mat B(N);
  Mat Ref(N);

  /* Fill A and B with deterministic values.  */
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      A(i, j) = 0.25 * (i + 1) - 0.5 * (j + 1);
      B(i, j) = 1.5 * (i + 1) + 0.125 * (j + 1);
      Ref(i, j) = A(i, j);
    }
  }

  /* Scalar reference: Ref += s * B */
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      Ref(i, j) += s * B(i, j);
    }
  }

  /* Test the dedicated fast path directly. */
  A.axpy_inplace(s, B);

  assert(A.rows() == N);
  assert(A.cols() == N);
  assert(A.num_elements() == Ref.num_elements());

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!almost_equal(A(i, j), Ref(i, j))) {
        std::fprintf(stderr, "Mismatch at (%d,%d): got %.17g expected %.17g\n",
                     i, j, A(i, j), Ref(i, j));
        assert(false);
      }
    }
  }

  /* Also test the high-level syntax if it is wired to the same specialized
   * path: A2 += s * B2
   */
  {
    Mat A2(N);
    Mat B2(N);
    Mat Ref2(N);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        A2(i, j) = -0.75 * (i + 1) + 0.25 * (j + 1);
        B2(i, j) = 0.50 * (i + 1) - 1.25 * (j + 1);
        Ref2(i, j) = A2(i, j);
      }
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        Ref2(i, j) += s * B2(i, j);
      }
    }

    A2 += s * B2;

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (!almost_equal(A2(i, j), Ref2(i, j))) {
          std::fprintf(
              stderr,
              "Operator+= mismatch at (%d,%d): got %.17g expected %.17g\n", i,
              j, A2(i, j), Ref2(i, j));
          assert(false);
        }
      }
    }
  }

  return 0;
#endif
}
