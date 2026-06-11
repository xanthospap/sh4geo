#include "gravity.hpp"
#include <array>
#include <cmath>
#ifdef DEBUG
#include <cassert>
#endif

namespace {
int sh2potential_impl(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, int max_degree,
    int max_order, double Re, double GM, double &U,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &M) noexcept {

  if (max_degree >=
      dso::detail::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS) {
    fprintf(stderr,
            "[ERROR] (Static) Size for NormalizedLegendreFactors must be "
            "augmented to perform computation (traceback: %s)\n",
            __func__);
    return 1;
  }

#ifdef DEBUG
  assert(cs.max_degree() >= cs.max_order());
  assert(max_degree <= cs.max_degree());
  assert(max_order <= cs.max_order());
  assert(W.rows() >= max_degree + 1);
  assert(W.cols() >= max_degree + 1);
  assert(M.rows() >= max_degree + 1);
  assert(M.cols() >= max_degree + 1);
#endif

  const int degree = max_degree;
  const int order = max_order;

  /* compute SH basis coefficients, Cnm->M and Snm->W */
  if (dso::gravity::sh_basis_cs_exterior(r / Re, degree, order, M, W)) {
    fprintf(stderr,
            "[ERROR] Failed computing spherical harmonics basis "
            "functions! (traceback: %s)\n",
            __func__);
    return 2;
  }

  double potential = 0e0;
  /* start from smaller terms */
  for (int m = order; m >= 1; --m) {
    const int row_idx = degree - m;
    const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx;
    const double *__restrict__ csSnm = cs.Snm().column(m) + row_idx;
    const double *__restrict__ Cnm = M.column(m) + row_idx; // M(m,m)
    const double *__restrict__ Snm = W.column(m) + row_idx; // W(m,m)
    for (int n = degree; n >= m; --n) {
      potential += (*csCnm) * (*Cnm) + (*csSnm) * (*Snm);
      /* update for next n */
      --csCnm;
      --csSnm;
      --Cnm;
      --Snm;
    } /* loop over n */
  } /* loop over m */

  /* special loop for m=0; here S(n,0) = 0 */
  {
    const int m = 0;
    const int row_idx = degree - m;
    const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx;
    const double *__restrict__ Cnm = M.column(m) + row_idx; // M(m,m)
    for (int n = degree; n >= m; --n) {
      potential += (*csCnm) * (*Cnm);
      /* update for next n */
      --csCnm;
      --Cnm;
    } /* loop over n */
  }

  /* scale and return */
  U = potential * GM / Re;
  return 0;
} /* end function */
} /* anonymous namespace */

int dso::sh2potential(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, double &U,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept {

  /* set (if needed) maximum degree and order of expansion */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = std::min(cs.max_order(), max_degree);
  if (max_order > max_degree) {
    fprintf(stderr,
            "[ERROR] Invalid degree/order for spherical harmonics expansion! "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  /* check computation degree and order w.r.t. the Stokes coeffs */
  if (max_degree > cs.max_degree()) {
    fprintf(stderr,
            "[ERROR] Requesting computing SH acceleration of degree %d, but "
            "Stokes coefficients are of size %dx%d (traceback: %s)\n",
            max_degree, cs.max_degree(), cs.max_order(), __func__);
    return 1;
  }
  if (max_order > cs.max_order()) {
    fprintf(stderr,
            "[ERROR] Requesting computing SH acceleration of order %d, but "
            "Stokes coefficients are of size %dx%d (traceback: %s)\n",
            max_order, cs.max_degree(), cs.max_order(), __func__);
    return 1;
  }

  /* set (if needed) geometric parameters of expansion */
  if (Re < 0)
    Re = cs.Re();
  if (GM < 0)
    GM = cs.GM();

  /* allocate (if needed) scratch space; for the scratch space we will be
   * computing terms up to [0, degree/order], hence our scratch matrices
   * should be able to hold degree/order + 1 elements.
   */
  int delete_mem_pool[] = {0, 0};
  if (!W) {
    W = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 1);
    delete_mem_pool[0] = 1;
  }
  if (!M) {
    M = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 1);
    delete_mem_pool[1] = 1;
  }

  /* check scratch space */
  if ((W->rows() < max_degree + 1) || (W->cols() < max_degree + 1) ||
      (M->rows() < max_degree + 1) || (M->cols() < max_degree + 1)) {
    fprintf(stderr,
            "[ERROR] Invalid size of mem pool for spherical harmonics "
            "expansion! (traceback: %s)\n",
            __func__);
    /* do not leak ... */
    if (delete_mem_pool[0])
      delete W;
    if (delete_mem_pool[1])
      delete M;
    return 1;
  }

  /* call core function */
  int status =
      sh2potential_impl(cs, r, max_degree, max_order, Re, GM, U, *W, *M);

  /* do we need to free memory ? */
  if (delete_mem_pool[0])
    delete W;
  if (delete_mem_pool[1])
    delete M;

  /* return */
  return status;
}