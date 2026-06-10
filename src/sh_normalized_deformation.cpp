
#include "gravity.hpp"
#include "load_love_numbers.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

int sh2deformation_impl(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, Eigen::Vector3d &dr,
    double gravity, double &potential, Eigen::Vector3d &potential_grad,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &M,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &W) noexcept {

  /* precomputed square roots and factors */
#ifdef PRECOMPUTED_SQRT_SHFACS
  const auto &Cf = dso::cunningham_weights();
#else
  const dso::detail::PrecomputedShSqrts psq =
      dso::detail::precomputed_sh_sqrts();
#endif

  /* effective degree and order */
  const int degree = max_degree;
  const int order = max_order;

  /* compute SH basis coefficients, Cnm->M and Snm->W */
  if (dso::gravity::sh_basis_cs_exterior(r / Re, degree + 1, order + 1, M, W)) {
    fprintf(stderr,
            "[ERROR] Failed computing spherical harmonics basis "
            "functions! (traceback: %s)\n",
            __func__);
    return 2;
  }

#ifndef PRECOMPUTED_SQRT_SHFACS
  /* 2^{1/2} */
  const double sqrt2 = std::sqrt(2e0);
#endif

  /*
   * gravity potential and acceleration per degree (n), i.e. V^{n} and \nabla
   * V^{n}
   */
  std::vector<double> potential_n(degree + 1, 0.0);
  std::vector<double> accel_xn(degree + 1, 0.0);
  std::vector<double> accel_yn(degree + 1, 0.0);
  std::vector<double> accel_zn(degree + 1, 0.0);

  const int mstart = order;
  const double *__restrict__ Mmm1 =
      (order >= 1) ? (M.column(mstart - 1)) : (nullptr);  // M(m-1,m-1)
  const double *__restrict__ Mmm0 = M.column(mstart - 0); // M(m,m)
  const double *__restrict__ Mmp1 = M.column(mstart + 1); // M(m+1,m+1)
  const double *__restrict__ Wmm1 =
      (order >= 1) ? (W.column(mstart - 1)) : (nullptr);  // W(m-1,m-1)
  const double *__restrict__ Wmm0 = W.column(mstart - 0); // W(m,m)
  const double *__restrict__ Wmp1 = W.column(mstart + 1); // W(m+1,m+1)

  /*
   * start from smaller terms. Note that for degrees m=0,1, we are using
   * seperate loops.
   */
  for (int m = order; m >= 2; --m) {
    const int row_idx = degree - m;
    const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx;
    const double *__restrict__ csSnm = cs.Snm().column(m) + row_idx;
#ifdef PRECOMPUTED_SQRT_SHFACS
    const double *__restrict__ d1wm1 = Cf.d1_wm1.column(m) + row_idx;
    const double *__restrict__ d1wm0 = Cf.d1_wm0.column(m) + row_idx;
    const double *__restrict__ d1wp1 = Cf.d1_wp1.column(m) + row_idx;
    auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
#else
    auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
#endif
    for (int n = degree; n >= m; --n) {
      const double cnm = *csCnm;
      const double snm = *csSnm;

      /* potential ... */
      potential_n[n] += (*csCnm) * (*Mmm0) + (*csSnm) * (*Wmm0);

#ifndef PRECOMPUTED_SQRT_SHFACS
      const double wm1 =
          std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2));
      const double wm0 =
          std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));
#else
      const double wm1 = (*d1wm1--);
      const double wm0 = (*d1wm0--);
      const double wp1 = (*d1wp1--);
#endif

      /* acceleration */
      int k = n + 2 - m;
      const double Cm1 = wm1 * Mmm1[k];
      const double Sm1 = wm1 * Wmm1[k];
      const double Cm0 = wm0 * Mmm0[k - 1];
      const double Sm0 = wm0 * Wmm0[k - 1];
      const double Cp1 = wp1 * Mmp1[k - 2];
      const double Sp1 = wp1 * Wmp1[k - 2];

      const double ax = cnm * (Cm1 - Cp1) + snm * (Sm1 - Sp1);
      const double ay = cnm * (-Sm1 - Sp1) + snm * (Cm1 + Cp1);
      const double az = cnm * (-2 * Cm0) + snm * (-2 * Sm0);

      const double scale_a = *fsqrt3;
      accel_xn[n] += scale_a * ax;
      accel_yn[n] += scale_a * ay;
      accel_zn[n] += scale_a * az;

      /* update for next n */
      --csCnm;
      --csSnm;
      --fsqrt3;
    } /* loop over n */

    /* kinda left-shift pointers for next iteration on new m <- m-1*/
    Mmp1 = Mmm0;
    Mmm0 = Mmm1;
    Wmp1 = Wmm0;
    Wmm0 = Wmm1;
    if (m > 2) {
      Mmm1 = M.column(m - 2); // M(m-1,m-1) ... but for new m = m - 1...
      Wmm1 = W.column(m - 2); // M(m-1,m-1)
    }
  } /* loop over m */

  /* order m = 1 (begin summation from smaller terms) */
  if (order >= 1) [[likely]] {
    int m = 1;
    const int row_idx1 = degree - m;
    const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx1;
    const double *__restrict__ csSnm = cs.Snm().column(m) + row_idx1;
#ifdef PRECOMPUTED_SQRT_SHFACS
    const double *__restrict__ d1wm1 = Cf.d1_wm1.column(m) + row_idx1;
    const double *__restrict__ d1wm0 = Cf.d1_wm0.column(m) + row_idx1;
    const double *__restrict__ d1wp1 = Cf.d1_wp1.column(m) + row_idx1;
    auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
#else
    auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
#endif
    for (int n = degree; n >= 1; --n) {
      const double cnm = *csCnm;
      const double snm = *csSnm;

      /* potential ... */
      potential_n[n] += (*csCnm) * (*Mmm0) + (*csSnm) * (*Wmm0);

      /* acceleration
       * only difference with the generalized formula (aka for random n,m)
       * is in wm1
       */
#ifndef PRECOMPUTED_SQRT_SHFACS
      const double wm1 =
          std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2)) * sqrt2;
      const double wm0 =
          std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));
#else
      const double wm1 = (*d1wm1--);
      const double wm0 = (*d1wm0--);
      const double wp1 = (*d1wp1--);
#endif

      const double Cm1 = wm1 * Mmm1[n + 2 - m];
      const double Sm1 = wm1 * Wmm1[n + 2 - m];
      const double Cm0 = wm0 * Mmm0[n + 1 - m];
      const double Sm0 = wm0 * Wmm0[n + 1 - m];
      const double Cp1 = wp1 * Mmp1[n - m];
      const double Sp1 = wp1 * Wmp1[n - m];

      const double ax = cnm * (Cm1 - Cp1) + snm * (Sm1 - Sp1);
      const double ay = cnm * (-Sm1 - Sp1) + snm * (Cm1 + Cp1);
      const double az = cnm * (-2e0 * Cm0) + snm * (-2 * Sm0);

      const double scale_a = *fsqrt3;
      accel_xn[n] += scale_a * ax;
      accel_yn[n] += scale_a * ay;
      accel_zn[n] += scale_a * az;
      /* update for next n */
      --csCnm;
      --csSnm;
      --fsqrt3;
    } /* loop over all n's for m=1 */
  }

  /* order m = 0; reset column pointers for M and W */
  const int m = 0;
  Mmm0 = M.column(0); // M(m,m)
  Mmp1 = M.column(1); // M(m+1,m+1)
  Wmm0 = W.column(0); // W(m,m)
  Wmp1 = W.column(1); // W(m+1,m+1)
  const int row_idx0 = degree - m;
  const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx0;
#ifdef PRECOMPUTED_SQRT_SHFACS
  const double *d1wm0 = Cf.d1_m0_wm0.data() + row_idx0;
  const double *d1wp1 = Cf.d1_m0_wp1.data() + row_idx0;
  auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
#else
  auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
#endif
  for (int n = degree; n >= 0; --n) {
    const double cnm = *csCnm;

    /* potential ... */
    potential_n[n] += (*csCnm) * (*Mmm0);

    /* acceleration */
#ifndef PRECOMPUTED_SQRT_SHFACS
    double wm0 = std::sqrt(static_cast<double>(n + 1) * (n + 1));
    double wp1 = std::sqrt(static_cast<double>(n + 1) * (n + 2)) / sqrt2;
#else
    const double wm0 = (*d1wm0--);
    const double wp1 = (*d1wp1--);
#endif

    double Cm0 = wm0 * Mmm0[n + 1];
    double Cp1 = wp1 * Mmp1[n];
    double Sp1 = wp1 * Wmp1[n];

    const double ax = cnm * (-2e0 * Cp1);
    const double ay = cnm * (-2e0 * Sp1);
    const double az = cnm * (-2e0 * Cm0);

    const double scale_a = *fsqrt3;
    accel_xn[n] += scale_a * ax;
    accel_yn[n] += scale_a * ay;
    accel_zn[n] += scale_a * az;
    /* update for next n */
    --csCnm;
    --fsqrt3;
  } /* loop over all n's for m=0 */

  /* unit vector at point (radial) */
  const Eigen::Vector3d u = r.normalized();

  /* Load Love numbers */
  const dso::LoadLoveNumbers &love = dso::groopsLoadLoveNumbers;

  /* (accumulated) deformation, potential gradient and potential */
  dr = Eigen::Vector3d::Zero();
  potential_grad = Eigen::Vector3d::Zero();
  potential = 0.0;

  Eigen::Vector3d an;
  for (int i = 0; i < degree + 1; i++) {
    an << accel_xn[i], accel_yn[i], accel_zn[i];
    an *= 1.0 / (2e0 * Re);
    dr += love.h[i] * potential_n[i] * u + love.l[i] * (an - an.dot(u) * u);
    potential_grad += an;
    potential += potential_n[i];
  }

  /* acceleration and potential in cartesian components */
  potential_grad *= (GM / Re);
  potential *= (GM / Re);
  dr /= (GM / Re / gravity);

  return 0;
}
} /* anonymous namespace */

int dso::sh2deformation(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Vector3d &dr, double gravity, double &potential,
    Eigen::Vector3d &potential_grad, int max_degree, int max_order, double Re,
    double GM,
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
   * computing terms up to [0, degree/order + 2], hence our scratch matrices
   * should be able to hold degree/order + 3 elements.
   */
  int delete_mem_pool[] = {0, 0};
  if (!W) {
    W = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2);
    delete_mem_pool[0] = 1;
  }
  if (!M) {
    M = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2);
    delete_mem_pool[1] = 1;
  }

  /* check scratch space */
  if ((W->rows() < max_degree + 2) || (W->cols() < max_degree + 2) ||
      (M->rows() < max_degree + 2) || (M->cols() < max_degree + 2)) {
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
      sh2deformation_impl(cs, r, dr, gravity, potential, potential_grad,
                          max_degree, max_order, Re, GM, *W, *M);

  /* do we need to free memory ? */
  if (delete_mem_pool[0])
    delete W;
  if (delete_mem_pool[1])
    delete M;

  /* return */
  return status;
}