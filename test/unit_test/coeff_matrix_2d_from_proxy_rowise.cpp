#include "coeff_matrix_2d.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <utility>
#include <random>

using namespace dso;
constexpr const double lower_bound = -1e3;
constexpr const double upper_bound = 1e3;
std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
std::default_random_engine re;

auto almost_equal = [](double a, double b) {
  return std::abs(a-b) < 1e-15 * (1.0 + std::abs(a) + std::abs(b));
};

/*
 * Matrix1 (rows=4, cols=4)
 *  1
 *  5  6
 *  9 10 11
 * 13 14 15 16
 */

int main() {
  CoeffMatrix2D<MatrixStorageType::RowWise> mat1(10, 10);

  /* fill in */
  for (int i = 0; i < mat1.rows(); i++) {
    for (int j = 0; j < mat1.cols(); j++) {
        mat1(i, j) = unif(re);
      }
    }

    auto expr = 2.0 * mat1.reduced_view(7,4);
    CoeffMatrix2D<MatrixStorageType::RowWise> B = expr;

    assert(B.rows()==7);
    assert(B.cols() == 4);

    for (int i = 0; i < B.rows(); i++) {
      for (int j = 0; j < B.cols(); j++) {
        assert(almost_equal(B(i,j), 2.0 * mat1(i,j)));
      }
    }

  return 0;
}
