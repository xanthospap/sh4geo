#include "gravity.hpp"
#include <cmath>
#include <cstdio>

int dso::gravity::sh_basis_cs_exterior(
    const Eigen::Vector3d &rsta, int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &S) noexcept {

  /* check requested max degree and order */
  if (max_degree < 0 || max_order < 0 || max_order > max_degree) {
    fprintf(
        stderr,
        "[Einv_r2OR] Invalid max degree/order for computing SH coefficients "
        "(traceback: %s)\n",
        __func__);
    return 1;
  }

  /* check size of input matrices matrices */
  if (C.rows() < max_degree + 1 || C.cols() < max_order + 1 ||
      S.rows() < max_degree + 1 || S.cols() < max_order + 1) {
    fprintf(stderr,
            "[Einv_r2OR] Invalid C/S size(s) for computing SH coefficients "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  /* check if we are ok with the NormalizedLegendreFactors size */
  constexpr int N =
      dso::detail::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;
  if (max_degree >= N || max_order >= N) {
    fprintf(stderr,
            "[Einv_r2OR] (Static) Size for NormalizedLegendreFactors must be "
            "augmented to perform computation (traceback: %s)\n",
            __func__);
    return 1;
  }

  /* Factors up to degree/order MAX_SIZE_FOR_ALF_FACTORS. */
  const dso::detail::NormalizedLegendreFactors &F =
      dso::detail::normalized_legendre_factors();

  /* (kinda) Associated Legendre Polynomials M and W
   * note that since we are going to compute derivatives (of the gravity
   * potential), we will compute M and W up to degree n+2,m+2
   */
  const double r2 = rsta.squaredNorm();
  if (!(r2 > 0.0)) {
    std::fprintf(stderr,
                 "[Einv_r2OR] Zero position vector in (traceback: %s)\n",
                 __func__);
    return 1;
  }
  const double inv_r = 1.0 / std::sqrt(r2);
  const double inv_r2 = 1.0 / r2;
  const double x = rsta.x() * inv_r2;
  const double y = rsta.y() * inv_r2;
  const double z = rsta.z() * inv_r2;

  /* start ALF iteration (can use scaling according to Holmes et al, 2002) */
  C(0, 0) = inv_r;
  S(0, 0) = 0.0;
  if (max_degree == 0) {
    return 0;
  }

  /* zero-th column of C */
  double *__restrict__ Ccol0 = C.column(0);
  /* zeroth-th column of associated Legendre factors (f1, f2) */
  const double *__restrict__ f10 = F.f1.column(0);
  const double *__restrict__ f20 = F.f2.column(0);

  /* first fill m=0 terms; note that W(n,0) = 0 (already set) */
  Ccol0[1] = f10[1] * z * Ccol0[0];

  for (int n = 2; n <= max_degree; n++) {
    Ccol0[n] = f10[n] * z * Ccol0[n - 1] + f20[n] * inv_r2 * Ccol0[n - 2];
  }

  /* just to be sure, zero-out first column of S, i.e. S(:,0)*/
  double *__restrict__ Scol0 = S.column(0);
  for (int n = 0; n <= max_degree; n++)
    Scol0[n] = 0e0;

  /* easy access to last used diagonal element C(n,n) and S(n,n) */
  double Cm1m1 = C(0, 0);
  double Sm1m1 = S(0, 0);

  /* fill all elements for order m >= 1 */
  for (int m = 1; m < max_order; m++) {
    /* m-th column of C and S */
    double *__restrict__ Cm0 = C.column(m); // Cm0 <- C(m,m)
    double *__restrict__ Sm0 = S.column(m); // Sm0 <- S(m,m)
    /* m-th column of associated Legendre factors (f1, f2) */
    const double *__restrict__ f1m = F.f1.column(m); // f1m <- F.f1(m,m)
    const double *__restrict__ f2m = F.f2.column(m); // f2m <- F.f2(m,m)

    /* M(m,m) and W(m,m) aka, diagonal */
    Cm0[0] = f1m[0] * (x * Cm1m1 - y * Sm1m1);
    Sm0[0] = f1m[0] * (y * Cm1m1 + x * Sm1m1);

    /* update last diagonal elements for next recursion */
    Cm1m1 = *Cm0;
    Sm1m1 = *Sm0;

    /* if n=m+1 , we do not have a M(n-2,...) aka sub-diagonal term */
    Cm0[1] = f1m[1] * z * Cm0[0];
    Sm0[1] = f1m[1] * z * Sm0[0];

    /* go on .... */
    for (int n = m + 2; n <= max_degree; n++) {
      const int nn = n - m;
      Cm0[nn] = f1m[nn] * z * Cm0[nn - 1] + f2m[nn] * inv_r2 * Cm0[nn - 2];
      Sm0[nn] = f1m[nn] * z * Sm0[nn - 1] + f2m[nn] * inv_r2 * Sm0[nn - 2];
    }
  }

  /* we do not need this block if (input) max_order=0 */
  if (max_order > 0) [[likely]] {
    /* well, we've left the lst column uncomputed */
    const int m = max_order;
    double *__restrict__ Cm0 = C.column(m);
    double *__restrict__ Sm0 = S.column(m);
    /* m-th column of associated Legendre factors (f1, f2) */
    const double *__restrict__ f1m = F.f1.column(m); // f1m <- F.f1(m,m)
    const double *__restrict__ f2m = F.f2.column(m); // f2m <- F.f2(m,m)

    Cm0[0] = f1m[0] * (x * Cm1m1 - y * Sm1m1);
    Sm0[0] = f1m[0] * (y * Cm1m1 + x * Sm1m1);

    if (max_degree > max_order) {
      Cm0[1] = f1m[1] * z * Cm0[0];
      Sm0[1] = f1m[1] * z * Sm0[0];
      for (int n = m + 2; n <= max_degree; n++) {
        const int nn = n - m;
        Cm0[nn] = f1m[nn] * z * Cm0[nn - 1] + f2m[nn] * inv_r2 * Cm0[nn - 2];
        Sm0[nn] = f1m[nn] * z * Sm0[nn - 1] + f2m[nn] * inv_r2 * Sm0[nn - 2];
      }
    }
  }

  return 0;
}