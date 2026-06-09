#include "gravity.hpp"
#include <array>
#include <cmath>
#ifdef DEBUG
#include <cassert>
#endif

namespace {

struct _PrecomputedSqrts {
  static const int N = dso::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;
  std::array<double, N> sqnp3;
  std::array<double, N> sqnp5;

  _PrecomputedSqrts() noexcept {
    for (int i = 0; i < N; i++) {
      sqnp3[i] = std::sqrt((double)(2 * i + 1) / (2 * i + 3));
      sqnp5[i] = std::sqrt((double)(2 * i + 1) / (2 * i + 5));
    }
  }
};

/** Spherical harmonics of Earth's gravity potential to acceleration and
 *  gradient using the algorithm due to Cunningham. The acceleration and
 *  gradient are computed in Cartesian components, i.e.
 *
 *  acceleration = (ax, ay, az)
 *
 * , and
 *
 *             | dax/dx dax/dy dax/dz |
 *  gradient = | day/dx day/dy day/dz |
 *             | daz/dx daz/dy daz/dz |
 *
 * @param[in] cs Normalized Stokes coefficients of spherical harmonics
 * @param[in] r  Position vector of satellite (aka point of computation) in
 *               an ECEF frame (e.g. ITRF)
 * @param[in] max_degree Max degree of spherical harmonics expansion
 * @param[in] max_order  Max order of spherical harmonics expansion
 * @param[in] Re Equatorial radius of the Earth
 * @param[in] GM Gravitational constant of Earth
 * @param[out] acc Acceleration in cartesian components in [m/s^2]
 * @param[out] gradient Gradient of acceleration in cartesian components
 * @param[in] W   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 * @param[in] M   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 */
int sh2gradient_cunningham_impl(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, int max_degree,
    int max_order, double Re, double GM, Eigen::Vector3d &acc,
    Eigen::Matrix<double, 3, 3> &gradient,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &M) noexcept {

  if (max_degree >
      dso::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS - 3) {
    fprintf(stderr,
            "[ERROR] (Static) Size for NormalizedLegendreFactors must be "
            "augmented to perform computation (traceback: %s)\n",
            __func__);
    return 1;
  }

  /* precomputed square roots */
#ifdef PRECOMPUTED_SQRT_SHFACS
  const auto &Cf = dso::cunningham_weights();
#else
  static const _PrecomputedSqrts psq;
#endif

#ifdef DEBUG
  assert(cs.max_degree() >= cs.max_order());
  assert(max_degree <= cs.max_degree());
  assert(max_order <= cs.max_order());
  assert(W.rows() >= max_degree + 3);
  assert(W.cols() >= max_degree + 3);
  assert(M.rows() >= max_degree + 3);
  assert(M.cols() >= max_degree + 3);
#endif

  /* effective degree and order */
  const int degree = max_degree;
  const int order = max_order;

  /* compute SH basis coefficients, Cnm->M and Snm->W */
  if (dso::gravity::sh_basis_cs_exterior(r / Re, degree + 2, order + 2, M, W)) {
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

  /* accumulated sums for acceleration and gradient components */
  double ax_sum = 0.0, ay_sum = 0.0, az_sum = 0.0;
  double gxx_sum = 0.0, gxy_sum = 0.0, gxz_sum = 0.0;
  double gyy_sum = 0.0, gyz_sum = 0.0, gzz_sum = 0.0;

  const int mstart = order;
  const double *__restrict__ Mmm2 =
      (order >= 2) ? (M.column(mstart - 2)) : (nullptr); // M(m-2,m-2)
  const double *__restrict__ Mmm1 =
      (order >= 1) ? (M.column(mstart - 1)) : (nullptr);  // M(m-1,m-1)
  const double *__restrict__ Mmm0 = M.column(mstart - 0); // M(m,m)
  const double *__restrict__ Mmp1 = M.column(mstart + 1); // M(m+1,m+1)
  const double *__restrict__ Mmp2 = M.column(mstart + 2); // M(m+2,m+2)
  const double *__restrict__ Wmm2 =
      (order >= 2) ? (W.column(mstart - 2)) : (nullptr); // W(m-2,m-2)
  const double *__restrict__ Wmm1 =
      (order >= 1) ? (W.column(mstart - 1)) : (nullptr);  // W(m-1,m-1)
  const double *__restrict__ Wmm0 = W.column(mstart - 0); // W(m,m)
  const double *__restrict__ Wmp1 = W.column(mstart + 1); // W(m+1,m+1)
  const double *__restrict__ Wmp2 = W.column(mstart + 2); // W(m+2,m+2)

  /* start from smaller terms. Note that for degrees m=0,1, we are using
   * seperate loops
   */
  for (int m = order; m >= 2; --m) {
    const int row_idx = degree - m;
    const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx;
    const double *__restrict__ csSnm = cs.Snm().column(m) + row_idx;
#ifdef PRECOMPUTED_SQRT_SHFACS
    const double *__restrict__ d1wm1 = Cf.d1_wm1.column(m) + row_idx;
    const double *__restrict__ d1wm0 = Cf.d1_wm0.column(m) + row_idx;
    const double *__restrict__ d1wp1 = Cf.d1_wp1.column(m) + row_idx;
    const double *__restrict__ d2wm2 = Cf.d2_wm2.column(m) + row_idx;
    const double *__restrict__ d2wm1 = Cf.d2_wm1.column(m) + row_idx;
    const double *__restrict__ d2wm0 = Cf.d2_wm0.column(m) + row_idx;
    const double *__restrict__ d2wp1 = Cf.d2_wp1.column(m) + row_idx;
    const double *__restrict__ d2wp2 = Cf.d2_wp2.column(m) + row_idx;
    auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
    auto fsqrt5 = Cf.grad_scale.cbegin() + degree;
#else
    auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
    auto fsqrt5 = psq.sqnp5.cbegin() + degree;
#endif
    for (int n = degree; n >= m; --n) {
      const double cnm = *csCnm;
      const double snm = *csSnm;
      /* acceleration */
      {
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
        ax_sum += scale_a * ax;
        ay_sum += scale_a * ay;
        az_sum += scale_a * az;
      }
      /* gradient */
      {
#ifndef PRECOMPUTED_SQRT_SHFACS
        const double wm2 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n - m + 3) * (n - m + 4)) *
                           ((m == 2) ? std::sqrt(2.0) : 1.0);
        const double wm1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n - m + 3) * (n + m + 1));
        const double wm0 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n + m + 1) * (n + m + 2));
        const double wp1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n + m + 1) * (n + m + 2) * (n + m + 3));
        const double wp2 = std::sqrt(static_cast<double>(n + m + 1) *
                                     (n + m + 2) * (n + m + 3) * (n + m + 4));
#else
        const double wm2 = (*d2wm2--);
        const double wm1 = (*d2wm1--);
        const double wm0 = (*d2wm0--);
        const double wp1 = (*d2wp1--);
        const double wp2 = (*d2wp2--);
#endif

        const double Cm2 = wm2 * Mmm2[n + 4 - m];
        const double Sm2 = wm2 * Wmm2[n + 4 - m];
        const double Cm1 = wm1 * Mmm1[n + 3 - m];
        const double Sm1 = wm1 * Wmm1[n + 3 - m];
        const double Cm0 = wm0 * Mmm0[n + 2 - m];
        const double Sm0 = wm0 * Wmm0[n + 2 - m];
        const double Cp1 = wp1 * Mmp1[n + 1 - m];
        const double Sp1 = wp1 * Wmp1[n + 1 - m];
        const double Cp2 = wp2 * Mmp2[n + 0 - m];
        const double Sp2 = wp2 * Wmp2[n + 0 - m];

        const double gxx =
            cnm * (Cm2 - 2 * Cm0 + Cp2) + snm * (Sm2 - 2 * Sm0 + Sp2);
        const double gxy = cnm * (-Sm2 + Sp2) + snm * (Cm2 - Cp2);
        const double gxz =
            cnm * (-2 * Cm1 + 2 * Cp1) + snm * (-2 * Sm1 + 2 * Sp1);
        const double gyy =
            cnm * (-Cm2 - 2 * Cm0 - Cp2) + snm * (-Sm2 - 2 * Sm0 - Sp2);
        const double gyz =
            cnm * (2 * Sm1 + 2 * Sp1) + snm * (-2 * Cm1 - 2 * Cp1);
        const double gzz = cnm * (4 * Cm0) + snm * (4 * Sm0);

        const double scale_g = *fsqrt5;
        gxx_sum += scale_g * gxx;
        gxy_sum += scale_g * gxy;
        gxz_sum += scale_g * gxz;
        gyy_sum += scale_g * gyy;
        gyz_sum += scale_g * gyz;
        gzz_sum += scale_g * gzz;
      }
      /* update for next n */
      --csCnm;
      --csSnm;
      --fsqrt3;
      --fsqrt5;
    } /* loop over n */

    /* kinda left-shift pointers for next iteration on new m <- m-1*/
    Mmp2 = Mmp1;
    Mmp1 = Mmm0;
    Mmm0 = Mmm1;
    Mmm1 = Mmm2;
    Wmp2 = Wmp1;
    Wmp1 = Wmm0;
    Wmm0 = Wmm1;
    Wmm1 = Wmm2;
    if (m > 2) {
      Mmm2 = M.column(m - 3); // M(m-2,m-2) ... but for new m = m - 1...
      Wmm2 = W.column(m - 3); // M(m-2,m-2)
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
    const double *__restrict__ d2wm1 = Cf.d2_wm1.column(m) + row_idx1;
    const double *__restrict__ d2wm0 = Cf.d2_wm0.column(m) + row_idx1;
    const double *__restrict__ d2wp1 = Cf.d2_wp1.column(m) + row_idx1;
    const double *__restrict__ d2wp2 = Cf.d2_wp2.column(m) + row_idx1;
    auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
    auto fsqrt5 = Cf.grad_scale.cbegin() + degree;
#else
    auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
    auto fsqrt5 = psq.sqnp5.cbegin() + degree;
#endif
    for (int n = degree; n >= 1; --n) {
      const double cnm = *csCnm;
      const double snm = *csSnm;
      {
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
        ax_sum += scale_a * ax;
        ay_sum += scale_a * ay;
        az_sum += scale_a * az;
      }
      /* gradient */
      {
#ifndef PRECOMPUTED_SQRT_SHFACS
        const double wm1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n - m + 3) * (n + m + 1)) *
                           sqrt2;
        const double wm0 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n + m + 1) * (n + m + 2));
        const double wp1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n + m + 1) * (n + m + 2) * (n + m + 3));
        const double wp2 = std::sqrt(static_cast<double>(n + m + 1) *
                                     (n + m + 2) * (n + m + 3) * (n + m + 4));
#else
        const double wm1 = (*d2wm1--);
        const double wm0 = (*d2wm0--);
        const double wp1 = (*d2wp1--);
        const double wp2 = (*d2wp2--);
#endif

        const double Cm1 = wm1 * Mmm1[n + 3 - m];
        const double Sm1 = wm1 * Wmm1[n + 3 - m];
        const double Cm0 = wm0 * Mmm0[n + 2 - m];
        const double Sm0 = wm0 * Wmm0[n + 2 - m];
        const double Cp1 = wp1 * Mmp1[n + 1 - m];
        const double Sp1 = wp1 * Wmp1[n + 1 - m];
        const double Cp2 = wp2 * Mmp2[n + 0 - m];
        const double Sp2 = wp2 * Wmp2[n + 0 - m];

        const double gxx = cnm * (-3 * Cm0 + Cp2) + snm * (-Sm0 + Sp2);
        const double gxy = cnm * (-Sm0 + Sp2) + snm * (-Cm0 - Cp2);
        const double gxz =
            cnm * (-2 * Cm1 + 2 * Cp1) + snm * (-2 * Sm1 + 2 * Sp1);
        const double gyy = cnm * (-Cm0 - Cp2) + snm * (-3 * Sm0 - Sp2);
        const double gyz = cnm * (2 * Sp1) + snm * (-2 * Cm1 - 2 * Cp1);
        const double gzz = cnm * (4 * Cm0) + snm * (4 * Sm0);

        const double scale_g = *fsqrt5;
        gxx_sum += scale_g * gxx;
        gxy_sum += scale_g * gxy;
        gxz_sum += scale_g * gxz;
        gyy_sum += scale_g * gyy;
        gyz_sum += scale_g * gyz;
        gzz_sum += scale_g * gzz;
      }
      /* update for next n */
      --csCnm;
      --csSnm;
      --fsqrt3;
      --fsqrt5;
    } /* loop over all n's for m=1 */
  }

  /* order m = 0; reset column pointers for M and W */
  const int m = 0;
  Mmm0 = M.column(0); // M(m,m)
  Mmp1 = M.column(1); // M(m+1,m+1)
  Mmp2 = M.column(2); // M(m+2,m+2)
  Wmm0 = W.column(0); // W(m,m)
  Wmp1 = W.column(1); // W(m+1,m+1)
  Wmp2 = W.column(2); // W(m+2,m+2)
  const int row_idx0 = degree - m;
  const double *__restrict__ csCnm = cs.Cnm().column(m) + row_idx0;
#ifdef PRECOMPUTED_SQRT_SHFACS
  auto d1wm0 = Cf.d1_m0_wm0.cbegin() + row_idx0;
  auto d1wp1 = Cf.d1_m0_wp1.cbegin() + row_idx0;
  auto d2wm0 = Cf.d2_m0_wm0.cbegin() + row_idx0;
  auto d2wp1 = Cf.d2_m0_wp1.cbegin() + row_idx0;
  auto d2wp2 = Cf.d2_m0_wp2.cbegin() + row_idx0;
  auto fsqrt3 = Cf.acc_scale.cbegin() + degree; /* square root factors */
  auto fsqrt5 = Cf.grad_scale.cbegin() + degree;
#else
  auto fsqrt3 = psq.sqnp3.cbegin() + degree; /* square root factors */
  auto fsqrt5 = psq.sqnp5.cbegin() + degree;
#endif
  for (int n = degree; n >= 0; --n) {
    const double cnm = *csCnm;
    {
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
      ax_sum += scale_a * ax;
      ay_sum += scale_a * ay;
      az_sum += scale_a * az;
    }
    /* gradient */
    {
#ifndef PRECOMPUTED_SQRT_SHFACS
      const double wm0 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2) * (n + 1) * (n + 2));
      const double wp1 =
          std::sqrt(static_cast<double>(n + 1) * (n + 1) * (n + 2) * (n + 3)) /
          sqrt2;
      const double wp2 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2) * (n + 3) * (n + 4)) /
          sqrt2;
#else
      const double wm0 = (*d2wm0--);
      const double wp1 = (*d2wp1--);
      const double wp2 = (*d2wp2--);
#endif

      const double Cm0 = wm0 * Mmm0[n + 2];
      const double Cp1 = wp1 * Mmp1[n + 1];
      const double Sp1 = wp1 * Wmp1[n + 1];
      const double Cp2 = wp2 * Mmp2[n + 0];
      const double Sp2 = wp2 * Wmp2[n + 0];

      const double gxx = cnm * (-2e0 * Cm0 + 2e0 * Cp2);
      const double gxy = cnm * (2e0 * Sp2);
      const double gxz = cnm * (4e0 * Cp1);
      const double gyy = cnm * (-2e0 * Cm0 - 2e0 * Cp2);
      const double gyz = cnm * (4e0 * Sp1);
      const double gzz = cnm * (4e0 * Cm0);

      const double scale_g = *fsqrt5;
      gxx_sum += scale_g * gxx;
      gxy_sum += scale_g * gxy;
      gxz_sum += scale_g * gxz;
      gyy_sum += scale_g * gyy;
      gyz_sum += scale_g * gyz;
      gzz_sum += scale_g * gzz;
    }
    /* update for next n */
    --csCnm;
    --fsqrt3;
    --fsqrt5;
  } /* loop over all n's for m=0 */

  /* acceleration and gradient in cartesian components */
  acc << ax_sum, ay_sum, az_sum;
  acc *= GM / (2e0 * Re * Re);
  gradient << gxx_sum, gxy_sum, gxz_sum, gxy_sum, gyy_sum, gyz_sum, gxz_sum,
      gyz_sum, gzz_sum;
  gradient *= GM / (4e0 * Re * Re * Re);

  return 0;
}
} /* unnamed namespace */

int dso::sh2gradient(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, 3> &gradient,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept {

  /* set (if needed) maximum degree and order of expansion */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = cs.max_order();
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
        max_degree + 3);
    delete_mem_pool[0] = 1;
  }
  if (!M) {
    M = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 3);
    delete_mem_pool[1] = 1;
  }

  /* check scratch space */
  if ((W->rows() < max_degree + 3) || (W->cols() < max_degree + 3) ||
      (M->rows() < max_degree + 3) || (M->cols() < max_degree + 3)) {
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
  int status = sh2gradient_cunningham_impl(cs, r, max_degree, max_order, Re, GM,
                                           acc, gradient, *W, *M);

  /* do we need to free memory ? */
  if (delete_mem_pool[0])
    delete W;
  if (delete_mem_pool[1])
    delete M;

  /* return */
  return status;
}
