#include "coeff_matrix_2d.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace dso;

namespace {

using Mat = CoeffMatrix2D<MatrixStorageType::LwTriangularColWise>;

/* Use a tolerance because the SIMD path may use FMA while the generic path
 * may evaluate as separate multiply/add operations. That can differ by a few
 * ulps.
 */
bool almost_equal(double a, double b, double eps = 1e-13) {
  return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}

/* Deterministic fill with nontrivial values. */
void fill_matrix(Mat &M, double alpha, double beta) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j <= i; ++j) {
      M(i, j) = alpha * static_cast<double>(i + 1) +
                beta * static_cast<double>(j + 1);
    }
  }
}

void check_same(const Mat &A, const Mat &B, const char *msg) {
  assert(A.rows() == B.rows());
  assert(A.cols() == B.cols());
  assert(A.num_elements() == B.num_elements());

  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!almost_equal(A(i, j), B(i, j))) {
        std::fprintf(stderr,
                     "%s mismatch at (%d,%d): got %.17g expected %.17g\n", msg,
                     i, j, A(i, j), B(i, j));
        assert(false);
      }
    }
  }
}

void run_case(int n, double s1, double s2) {
  Mat A0(n);
  Mat A_direct(n);
  Mat A_expr(n);
  Mat A_ref(n);
  Mat B1(n);
  Mat B2(n);

  fill_matrix(A0, 0.25, -0.50);
  fill_matrix(B1, 1.50, 0.125);
  fill_matrix(B2, -0.75, 0.375);

  A_direct = A0;
  A_expr = A0;
  A_ref = A0;

  /* Scalar reference. */
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      A_ref(i, j) += s1 * B1(i, j) + s2 * B2(i, j);
    }
  }

  /* Explicit optimized call. */
  A_direct.axpy2_inplace(s1, B1, s2, B2);

  /* Generic overloaded expression path. */
  A_expr += s1 * B1 + s2 * B2;

  /* Check dimensions preserved. */
  assert(A_direct.rows() == n);
  assert(A_direct.cols() == n);
  assert(A_expr.rows() == n);
  assert(A_expr.cols() == n);

  /* Compare both implementations against scalar reference. */
  check_same(A_direct, A_ref, "axpy2_inplace");
  check_same(A_expr, A_ref, "operator+= expr");

  /* And compare them directly to each other. */
  check_same(A_direct, A_expr, "direct vs expr");
}

} // namespace

int main() {
  /* Choose sizes that exercise different vector/tail cases:
   * n*(n+1)/2 gives element counts with different mod 8 behavior.
   */
  run_case(1, 1.25, -0.375);
  run_case(2, -1.75, 0.625);
  run_case(3, 0.50, 1.125);
  run_case(4, -0.75, -0.25);
  run_case(5, 1.875, 0.375);
  run_case(7, -1.25, 0.50);
  run_case(8, 0.625, -1.50);
  run_case(9, -0.875, 1.25);
  run_case(15, 1.50, -0.625);
  run_case(16, -1.125, 0.875);
  run_case(17, 0.75, -1.375);
  run_case(22, -1.75, 0.625);
  run_case(40, 1.25, -0.375);
  run_case(100, -0.5, 1.75);
  run_case(200, 0.875, -1.125);

  return 0;
}