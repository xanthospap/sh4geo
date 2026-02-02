#include "coeff_matrix_2d.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <utility>

using namespace dso;

int main() {
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat1(599, 599);
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat2(599, 599);

  /* fill in */
  int k = 0;
  for (int i = 0; i < mat1.rows(); i++) {
    for (int j = 0; j <= i; j++) {
      mat1(i, j) = (double)(++k);
      mat2(i, j) = (double)(++k) * .1;
    }
  }

  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat3 =
      1e0 * mat1.reduced_view(499, 499) + 2e0 * mat2.reduced_view(499, 499);
  assert(mat3.rows() == 499);
  assert(mat3.cols() == 499);
  for (int i = 0; i < 499; i++) {
    for (int j = 0; j <= i; j++) {
      assert(mat3(i, j) == 1e0 * mat1(i, j) + 2e0 * mat2(i, j));
    }
  }

  mat3 += 1e0 * mat1.reduced_view(399, 399) + 2e0 * mat2.reduced_view(399, 399);
  assert(mat3.rows() == 499);
  assert(mat3.cols() == 499);
  for (int i = 0; i < 499; i++) {
    for (int j = 0; j <= i; j++) {
      if (i < 399 && j < 399)
        assert(mat3(i, j) == 2e0 * (1e0 * mat1(i, j) + 2e0 * mat2(i, j)));
      else
        assert(mat3(i, j) == 1e0 * (1e0 * mat1(i, j) + 2e0 * mat2(i, j)));
    }
  }

  return 0;
}
