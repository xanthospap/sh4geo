/** @file
 *  @brief Internal spherical-harmonic helper structures and low-level
 *         implementation utilities.
 *
 *  This header collects the internal building blocks used by the public
 *  spherical-harmonic gravity routines.
 *
 *  It contains:
 *
 *  - recurrence factors for the construction of fully normalized exterior
 *    spherical-harmonic basis functions,
 *  - degree-only square-root factors used in derivative formulas,
 *  - precomputed Cunningham coefficients for first and second Cartesian
 *    derivatives of the potential,
 *  - and other low-level helpers shared by the gravity, gradient, potential,
 *    and deformation kernels.
 *
 * The quantities represented here are not, by themselves, user-facing
 * results. Instead, they are fixed normalization-dependent coefficients that
 * depend only on degree/order and are reused repeatedly by the hot
 * spherical-harmonic evaluation routines.
 *
 *  The helper structures in this file assume the standard geodetic
 *  **fully normalized** spherical-harmonic convention:
 *
 *  - fully normalized Stokes coefficients
 *    \f$\bar{C}_{nm}, \bar{S}_{nm}\f$,
 *  - fully normalized associated Legendre / solid harmonic basis functions,
 *  - and the corresponding normalized derivative formulas.
 *
 *  @note
 *  This header is primarily implementation-oriented. Public application code
 *  will normally interact with the higher-level routines declared in
 *  `gravity.hpp`, while this file provides the reusable internal machinery.
 */

#ifndef DSO_GEOPOTENTIAL_ACCELERATION_DETAILS_HPP
#define DSO_GEOPOTENTIAL_ACCELERATION_DETAILS_HPP

#include "eigen3/Eigen/Eigen"
#include "stokes_coefficients.hpp"
#include <array>

namespace dso {

namespace detail {

/** @brief Precomputed recurrence coefficients for normalized exterior solid
 *         spherical harmonics.
 *
 *  This helper stores the degree/order-dependent coefficients required by
 *  the recurrence used in @ref dso::gravity::sh_basis_cs_exterior to build
 *  the real exterior spherical-harmonic basis functions
 *  @f$C_{nm}@f$ and @f$S_{nm}@f$.
 *
 *  The stored arrays are:
 *
 *  - @ref f1 :
 *    the coefficient multiplying the degree-`n-1` term in the recurrence,
 *  - @ref f2 :
 *    the coefficient multiplying the degree-`n-2` term in the recurrence,
 *  - @ref f3 :
 *    the degree-only factor
 *    @f[
 *      \sqrt{\frac{2n+1}{2n+3}},
 *    @f]
 *    used later in the Cunningham first-derivative formulas for
 *    gravitational acceleration and deformation.
 *
 *  The recurrence coefficients depend only on the selected normalization and
 *  on the indices @f$(n,m)@f$; they do not depend on the evaluation point.
 *  They are therefore ideal candidates for one-time precomputation and
 *  repeated reuse in the hot spherical-harmonic kernels.
 *
 *  Storage is organized using
 *  @ref dso::MatrixStorageType::LwTriangularColWise because the admissible
 *  degree/order domain is triangular:
 *  @f[
 *    0 \le m \le n.
 *  @f]
 *
 *  @note
 *  The maximum supported degree/order is controlled by
 *  @ref MAX_SIZE_FOR_ALF_FACTORS. If a larger model is needed, this
 * compile-time limit must be increased consistently with the corresponding
 * implementation.
 *
 *  @see dso::gravity::sh_basis_cs_exterior
 */
struct NormalizedLegendreFactors {
  /** @brief Maximum supported degree/order for the precomputed recurrence
   *         factors.
   */
  static constexpr const int MAX_SIZE_FOR_ALF_FACTORS = 201;

  /** @brief Recurrence coefficient multiplying the degree-`n-1` term. */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f1;

  /** @brief Recurrence coefficient multiplying the degree-`n-2` term. */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f2;

  /** @brief Degree-only factor
   *         @f$\sqrt{(2n+1)/(2n+3)}@f$ used in first-derivative formulas.
   */
  std::array<double, MAX_SIZE_FOR_ALF_FACTORS> f3;

  /** @brief Construct and fill all recurrence factors.
   *
   *  This constructor performs the one-time initialization of @ref f1,
   *  @ref f2, and @ref f3.
   */
  NormalizedLegendreFactors() noexcept;
};

const NormalizedLegendreFactors &normalized_legendre_factors() noexcept;

/** @brief Degree-only square-root factors used in spherical-harmonic
 *         derivative formulas.
 *
 *  This helper stores degree-dependent factors that occur repeatedly in
 *  Cunningham-style Cartesian derivatives of the fully normalized exterior
 *  spherical-harmonic potential.
 *
 *  These factors depend only on the degree @f$n@f$, not on the order
 *  @f$m@f$ and not on the evaluation point. They are therefore ideal for
 *  one-time precomputation and repeated reuse.
 *
 *  The stored arrays are:
 *
 *  - @ref sqnp3 \f$\sqrt{\frac{2n+1}{2n+3}} \f$ used in first-derivative
 * formulas, i.e. gravitational acceleration and potential-gradient
 * computations,
 *
 *  - @ref sqnp5 \f$\sqrt{\frac{2n+1}{2n+5}}\f$ used in second-derivative
 * formulas, i.e. gravity-gradient tensor computations.
 *
 *  These factors are part of the fully normalized spherical-harmonic
 *  formulation commonly used in geodesy.
 *
 *  @see dso::sh2gravity
 *  @see dso::sh2gradient
 */
struct PrecomputedShSqrts {
  /** @brief Number of precomputed degree entries. */
  static const int N = NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;

  /** @brief Stores @f$\sqrt{(2n+1)/(2n+3)}@f$. */
  std::array<double, N> sqnp3;

  /** @brief Stores @f$\sqrt{(2n+1)/(2n+5)}@f$. */
  std::array<double, N> sqnp5;

  /** @brief Construct and fill all degree-only square-root factors. */
  PrecomputedShSqrts() noexcept;
};
const PrecomputedShSqrts &precomputed_sh_sqrts() noexcept;

#ifdef PRECOMPUTED_SQRT_SHFACS
/** @brief Precomputed degree/order-dependent coefficients for Cunningham-style
 *         Cartesian derivatives of the spherical-harmonic potential.
 *
 *  This helper stores the fixed numerical factors that appear in the
 *  Cunningham formulas for:
 *
 *  - first Cartesian derivatives of the potential
 *    (gravity / potential gradient),
 *  - second Cartesian derivatives of the potential
 *    (gravity-gradient tensor),
 *  - and the first-derivative terms reused in the deformation routine.
 *
 *  The purpose of this structure is to remove repeated square-root
 *  evaluations from the hot derivative kernels. Since all these factors
 *  depend only on degree/order and normalization, they can be computed once
 *  and reused for every evaluation point.
 *
 *  The stored data are split into two groups:
 *
 *  1. **Degree-only arrays**
 *     for factors that depend only on @f$n@f$, such as
 *     @ref acc_scale and @ref grad_scale, together with special zonal
 *     (`m = 0`) coefficients used in the dedicated zonal branches.
 *
 *  2. **Lower-triangular column-wise matrices**
 *     for coefficients that depend on both degree and order,
 *     used in the general Cunningham formulas for
 *     @f$m \ge 1@f$ or @f$m \ge 2@f$.
 *
 *  The naming convention is:
 *
 *  - `d1_*` : first-derivative coefficients,
 *  - `d2_*` : second-derivative coefficients,
 *  - `wm*`  : coefficients multiplying lower-order / lower-index neighboring
 *             terms,
 *  - `wp*`  : coefficients multiplying higher-order / higher-index neighboring
 *             terms.
 *
 *  @note
 *  This helper is only available when `PRECOMPUTED_SQRT_SHFACS` is enabled.
 *
 *  @see dso::sh2gravity
 *  @see dso::sh2gradient
 *  @see dso::sh2deformation
 */
struct CunninghamWeights {
  /** @brief Maximum supported degree/order. */
  static constexpr int MAX_N =
      NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;

  /** @name Degree-only factors
   *  @{
   */

  /** @brief Stores @f$\sqrt{(2n+1)/(2n+3)}@f$ for first-derivative formulas. */
  std::array<double, MAX_N> acc_scale;

  /** @brief Stores @f$\sqrt{(2n+1)/(2n+5)}@f$ for second-derivative formulas.
   */
  std::array<double, MAX_N> grad_scale;

  /** @brief Zonal (`m=0`) first-derivative coefficient multiplying
   *         the `m=0` neighbor.
   */
  std::array<double, MAX_N> d1_m0_wm0;

  /** @brief Zonal (`m=0`) first-derivative coefficient multiplying
   *         the `m=1` neighbor.
   */
  std::array<double, MAX_N> d1_m0_wp1;

  /** @brief Zonal (`m=0`) second-derivative coefficient for the central term.
   */
  std::array<double, MAX_N> d2_m0_wm0;

  /** @brief Zonal (`m=0`) second-derivative coefficient multiplying
   *         the `m=1` neighbor.
   */
  std::array<double, MAX_N> d2_m0_wp1;

  /** @brief Zonal (`m=0`) second-derivative coefficient multiplying
   *         the `m=2` neighbor.
   */
  std::array<double, MAX_N> d2_m0_wp2;

  /** @} */

  /** @name First-derivative coefficients
   *  @{
   */

  /** @brief General first-derivative coefficient associated with the
   *         `(n+1,m-1)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wm1;

  /** @brief General first-derivative coefficient associated with the
   *         `(n+1,m)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wm0;

  /** @brief General first-derivative coefficient associated with the
   *         `(n+1,m+1)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wp1;

  /** @} */

  /** @name Second-derivative coefficients
   *  @{
   */

  /** @brief General second-derivative coefficient associated with the
   *         `(n+2,m-2)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm2;

  /** @brief General second-derivative coefficient associated with the
   *         `(n+2,m-1)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm1;

  /** @brief General second-derivative coefficient associated with the
   *         `(n+2,m)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm0;

  /** @brief General second-derivative coefficient associated with the
   *         `(n+2,m+1)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wp1;

  /** @brief General second-derivative coefficient associated with the
   *         `(n+2,m+2)` neighbor.
   */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wp2;

  /** @} */

  /** @brief Construct and fill all precomputed Cunningham coefficients. */
  CunninghamWeights();
};
const CunninghamWeights &cunningham_weights() noexcept;
#endif

} /* namespace detail */

namespace gravity {

/** @brief  Compute the spherical harmonic basis functions Cnm, Snm.
 *
 * This function computes the spherical harmonic basis functions Cnm, Snm up
 * to a specified degree n, and order m, evaluated at a 3D point in space.
 *
 * These functions are the real-valued solid spherical harmonics, commonly
 * used in geodesy and gravity field modeling. To be a bit more rigorous, these
 * are real exterior solid spherical harmonic basis functions, using fully
 * normalized associated Legendre functions.
 *
 * Computes the real solid spherical harmonics (4π normalized):
 * Cnm = (1/r**(n+1)) cos(mλ) Pnm(cosθ)
 * Snm = (1/r**(n+1)) sin(mλ) Pnm(cosθ)
 *
 * At the end of the function
 * C(n,m) holds the real-valued cosine basis Cnm
 * S(n,m) holds the real-valued sine basis Snm
 *
 * These can then be multiplied by the corresponding cnm, snm gravity field
 * coefficients to compute potential, acceleration, etc.
 *
 * @param[in] rsta The point of computation; should be exterior to Earth,
 * given in geocentric cartesian coordinates, ECEF [m]. in gravity-field/geodesy
 * use, this function is normally called with rsta = r / Re.
 * @param[in] max_degree Max degree of computation.
 * @param[in] max_order  Max order of computation.
 * @param[in] C    A lower triangular, column-wise matrix where the Cnm
 *                 coefficients are stored after computation. Its size should
 *                 be large enough to hold the computed coefficients, i.e.
 *                 (C.rows() >= max_degree+1) && (C.cols() >= max_degree+1).
 * @param[in] S    A lower triangular, column-wise matrix where the Snm
 *                 coefficients are stored after computation. Its size should
 *                 be large enough to hold the computed coefficients, i.e.
 *                 (S.rows() >= max_degree+1) && (S.cols() >= max_degree+1).
 * @return         Anything other than zero denotes an error.
 *
 * @note
 * This routine only guarantees the values in the triangular region
 * `0 <= m <= max_order`, `m <= n <= max_degree`.
 * Values outside that region are left unspecified.
 */
[[nodiscard]]
int sh_basis_cs_exterior(
    const Eigen::Vector3d &rsta, int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &S) noexcept;
} /* namespace gravity */

} /* namespace dso */

#endif