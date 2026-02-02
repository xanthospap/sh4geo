#include "coeff_matrix_2d.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <utility>

using namespace dso;

int main() {
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat1(4, 4);
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat2(5, 4);
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat3(4, 4);
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> mat4(4, 4);

  /* this should NOT be allowed to compile cause mat2 is of different type */
  mat4 = mat1 + .1 * mat2 + .2 * mat3;

  /* all done */
  return 0;
}
