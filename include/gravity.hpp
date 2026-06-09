/** @file
 * Computation of gravity acceleration using a geopotential model.
 */

#ifndef __DSO_GEOPOTENTIAL_ACCELERATAION_HPP__
#define __DSO_GEOPOTENTIAL_ACCELERATAION_HPP__

#include "eigen3/Eigen/Eigen"
#include "stokes_coefficients.hpp"
// #include "geodesy/transformations.hpp"
// #include "iers2010/load_love_numbers.hpp"
// #include "iers2010/stokes_coefficients.hpp"

namespace dso {

struct NormalizedLegendreFactors {
  /* Max size for ALF factors; if degree is more than this, then it must
   * be augmented. For now, OK
   */
  static constexpr const int MAX_SIZE_FOR_ALF_FACTORS = 201;

  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f1;
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f2;
  std::array<double, MAX_SIZE_FOR_ALF_FACTORS> f3;

  /** @brief Precomputation of recurrence coefficients for normalized
   * Legendre-based exterior solid harmonics.
   *
   *  This translation unit initializes @ref dso::NormalizedLegendreFactors,
   *  a small helper object that stores recurrence coefficients reused by the
   *  spherical-harmonic basis generator
   *  @ref dso::gravity::sh_basis_cs_exterior.
   *
   *  The stored factors are:
   *
   *  - `f1(n,m)` : the coefficient multiplying the previous-degree term
   *  - `f2(n,m)` : the coefficient multiplying the degree-`n-2` term
   *  - `f3[n]`   : the auxiliary factor
   *      @f$\sqrt{(2n+1)/(2n+3)}@f$, used later in acceleration formulas
   *
   *  ## Why precompute these factors?
   *
   * The basis-generation and gravity-evaluation routines are called many
   * times, often in tight loops over many spacecraft positions or grid points.
   * The recurrence coefficients depend only on `(n,m)` and the chosen
   * normalization, not on the evaluation point. Therefore they are ideal
   * candidates for a one-time precomputation.
   *
   * This constructor is intentionally written for correctness and clarity
   * rather than micro-optimization: it runs once, whereas the basis-generation
   * kernels may run extremely often.
   */
  NormalizedLegendreFactors() noexcept;
}; /* struct NormalizedLegendreFactors */

#ifdef PRECOMPUTED_SQRT_SHFACS
struct CunninghamWeights {
  static constexpr int MAX_N =
      NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;

  /* degree-only */
  std::array<double, MAX_N> acc_scale;
  std::array<double, MAX_N> grad_scale;
  std::array<double, MAX_N> d1_m0_wm0;
  std::array<double, MAX_N> d1_m0_wp1;
  std::array<double, MAX_N> d2_m0_wm0;
  std::array<double, MAX_N> d2_m0_wp1;
  std::array<double, MAX_N> d2_m0_wp2;

  /* first derivatives: gravity + deformation */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wm1;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wm0;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d1_wp1;

  /* second derivatives: gravity gradient */
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm2;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm1;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wm0;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wp1;
  CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> d2_wp2;

  CunninghamWeights();
};
const CunninghamWeights &cunningham_weights() noexcept;
#endif

/** Acceleration due to point mass at r_cb on a mass at r.
 *
 * Compute the aceleration induced on a body at the position vector r,
 * caused by a point mass with gravitational constant GM at position rcb.
 * This is normally used to compute third-body acceleration on a
 * satellite, induced by e.g. the Sun or Moon.
 *
 * @param[in] r Geocentric position of attracted mass, i.e. satellite in
 * [m]. The vector holds Cartesian components, i.e. r=(x,y,z).
 * @param[in] rcb Geocentric position vector of the attracting body, i.e.
 * the 'third body' (e.g. Sun or Moon) in [m].
 * @param[in] GMcb gravitational constant of the 'third body', i.e. G *
 * M_cb, in [m^3/ sec^2]
 * @return Acceleration induced on the mass, in Cartesian components, in
 *         units of [m/s^2]. I.e. a = (a_x, a_y, a_z)
 *
 * @note Instead of using [m] as input units, [km] can also be used, as
 * long as it is used consistently for ALL inputs (r, rcb and GM). in this
 *       case, the resulting acceleration will be given in units of
 * [km/s^2].
 */
// Eigen::Matrix<double, 3, 1>
// point_mass_acceleration(const Eigen::Matrix<double, 3, 1> &r,
//                         const Eigen::Matrix<double, 3, 1> &rcb,
//                         double GMcb) noexcept;

/** Acceleration due to point mass at r_cb on a mass at r.
 *
 * Same as above, only in this case we also compute and return the Jacobian
 * matrix da/dr, i.e.
 *
 *     | dax/dx dax/dy dax/dz |
 * J = | day/dx day/dy day/dz |
 *     | daz/dx daz/dy daz/dz |
 *
 * @param[in] r Geocentric position of attracted mass, i.e. satellite in [m].
 *              The vector holds Cartesian components, i.e. r=(x,y,z).
 * @param[in] rcb Geocentric position vector of the attracting body, i.e. the
 *              'third body' (e.g. Sun or Moon) in [m].
 * @param[in] GMcb gravitational constant of the 'third body', i.e. G * M_cb,
 *              in [m^3/ sec^2]
 * @param[out] jacobian The Jacobian 3x3 matrix da/dr
 * @return Acceleration induced on the mass, in Cartesian components, in
 *         units of [m/s^2]. I.e. a = (a_x, a_y, a_z)
 *
 * @note Instead of using [m] as input units, [km] can also be used, as long
 *       as it is used consistently for ALL inputs (r, rcb and GM). in this
 *       case, the resulting acceleration will be given in units of [km/s^2].
 */
// Eigen::Matrix<double, 3, 1>
// point_mass_acceleration(const Eigen::Matrix<double, 3, 1> &r,
//                         const Eigen::Matrix<double, 3, 1> &rcb, double GMcb,
//                         Eigen::Matrix<double, 3, 3> &jacobian) noexcept;

/** Spherical harmonics of Earth's gravity potential to acceleration and
 *  gradient using the algorithm due to Cunningham. The acceleration and
 *  gradient are computed in Cartesian components, i.e.
 *
 *  acceleration = (ax, ay, az), and
 *
 *             | dax/dx dax/dy dax/dz |
 *  gradient = | day/dx day/dy day/dz |
 *             | daz/dx daz/dy daz/dz |
 *
 * @param[in] cs Normalized Stokes coefficients of spherical harmonics
 * @param[in] r  Position vector of satellite (aka point of computation) in
 *               an ECEF frame (e.g. ITRF)
 * @param[in] max_degree Max degree of spherical harmonics expansion. If set
 *               to a negative number, the degree of expansion will be derived
 *               from the cs input parameter, i.e. cs.max_degree()
 * @param[in] max_order  Max order of spherical harmonics expansion. If set
 *               to a negative number, the order of expansion will be derived
 *               from the cs input parameter, i.e. cs.max_order()
 * @param[in] Re Equatorial radius of the Earth in [m]. If set to a negative
 *               number, then the cs.Re() method will be used to get it.
 * @param[in] GM Gravitational constant of Earth. If set to a negative
 *               number, then the cs.GM() method will be used to get it.
 * @param[out] acc Acceleration in cartesian components in [m/s^2]
 * @param[out] gradient Gradient of acceleration in cartesian components
 * @param[in] W   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2).
 *                If not given, then the function will allocate and free the
 *                required memmory.
 * @param[in] M   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 *                If not given, then the function will allocate and free the
 *                required memmory.
 * @return        Anything other than zero denotes an error.
 */
[[nodiscard]]
int sh2gradient(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, 3> &gradient,
    int max_degree = -1, int max_order = -1, double Re = -1, double GM = -1,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W =
        nullptr,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M =
        nullptr) noexcept;

[[nodiscard]]
int sh2potential(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    double &U, int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept;

/** @brief Compute surface displacement due to a surface load.
 *
 * Computes the 3D displacement vector, dr, of a point on the Earth's surface
 * due to surface mass loading, expressed in an Earth-centered coordinate
 * frame.
 *
 * dr = dr_radial + dr_tangential
 * dr_radial     = Σ h(n)/g * V(n) * r
 * dr_tangential = Σ l(n)/g * (grad V(n) - <grad V(n), r> * r)
 *
 * dr       : Total displacement vector at the observation point [m]
 * V(n)     : Load potential at spherical harmonic degree n
 * grad V(n): Gradient (vector) of V(n) at the observation point
 * g        : Local acceleration due to gravity
 * h(n),l(n): Load Love numbers for degree n (for vertical and horizontal
 *            deformation respectively)
 * r        : Local upward (radial) unit vector
 *
 * @param[in] point A point on or outside the Earth, specified by its
 *            geocentric cartesian coordinates (ITRF) [m]
 * @param[in] cs Stokes coefficients of the surface load, the 'source' of the
 *            displacement
 * @param[out] dr Displacement vector at the point of interest, in cartesian
 *            coordiantes [m]. Hence, the point will have moved to
 *            new_location = point + dr
 * @param[out] gravity Gravity acceleration at the point of interest (since we
 *            are computing it, we might as well return it!) in [m/s^2]
 * @param[out] potential The (total) gravitational potential at the point of
 *            interest (since we are computing it, we might as well return
 *            it!) in [m/s^2]
 * @param[in] max_degree Max degree of spherical harmonics expansion. Should
 *            be max_degree <= cs.max_degree().
 * @param[in] max_order Max order of spherical harmonics expansion. Should
 *            be max_order <= cs.max_order() and max_order <= max_degree.
 * @param[inout] M Scratch space to be used by the function. The matrix should
 *            have size at least >= max_degree+2
 * @param[inout] W Scratch space to be used by the function. The matrix should
 *            have size at least >= max_degree+2
 * @return Anything other than zero denotes an error.
 */
[[nodiscard]]
int sh2deformation(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Vector3d &dr, Eigen::Vector3d &gravity, double &potential,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept;

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

/** @brief Computes the 3D displacement vector due to surface mass loading.
 *
 * This is just a wrapper for dso::gravity::sh_deformation
 *
 * @param[in] point A point on or outside the Earth, specified by its
 *            geocentric cartesian coordinates (ITRF) [m]
 * @param[in] cs Stokes coefficients of the surface load, the 'source' of the
 *            displacement
 * @param[out] dr Displacement vector at the point of interest, in cartesian
 *            coordiantes [m]. Hence, the point will have moved to
 *            new_location = point + dr
 * @param[out] gravity Gravity acceleration at the point of interest (since we
 *            are computing it, we might as well return it!) in [m/s^2]
 * @param[out] potential The (total) gravitational potential at the point of
 *            interest (since we are computing it, we might as well return
 *            it!) in [m/s^2]
 * @param[in] max_degree Max degree of spherical harmonics expansion. Should
 *            be max_degree <= cs.max_degree(). If set to a negative number,
 *            cs.max_degree() will be used.
 * @param[in] max_order Max order of spherical harmonics expansion. Should
 *            be max_order <= cs.max_order() and max_order <= max_degree. If
 *            set to a negative number, cs.max_order() will be used.
 * @param[inout] M Scratch space to be used by the function. The matrix should
 *            have size at least >= max_degree+2. If not provided, it will be
 *            allocated.
 * @param[inout] W Scratch space to be used by the function. The matrix should
 *            have size at least >= max_degree+2. If not provided, it will be
 *            allocated.
 * @return Anything other than zero denotes an error.
 */
//[[nodiscard]] int sh_deformation(
//    const Eigen::Vector3d &rsta, const dso::StokesCoeffs &cs,
//    Eigen::Vector3d &gravity, double &potential, Eigen::Vector3d &dr,
//    int max_degree = -1, int max_order = -1,
//    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M =
//        nullptr,
//    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W =
//        nullptr) noexcept;

/** @brief Compute the gravitational potential from normalized Stokes
 * coefficients.
 *
 * Computes the scalar gravitational potential at an Earth-fixed Cartesian point
 * using the spherical-harmonic model stored in `cs`.
 *
 * The input point must be expressed in the same terrestrial frame as the
 * geopotential coefficients, typically ITRF/ECEF.
 *
 * The returned potential has units of:
 *
 *     m^2 / s^2
 *
 * The computation uses normalized exterior spherical-harmonic basis functions
 * and column-major lower-triangular coefficient storage.
 *
 * @param[in]  cs
 *     Spherical-harmonic/Stokes coefficient set. Supplies `Cnm`, `Snm`,
 *     reference radius, gravitational constant, and maximum degree/order.
 *
 * @param[in]  point
 *     Earth-fixed Cartesian position vector, in metres, where the potential is
 *     evaluated.
 *
 * @param[out] potential
 *     Computed gravitational potential at `point`, in m^2/s^2.
 *
 * @param[in]  max_degree
 *     Maximum spherical-harmonic degree to use. If negative, the maximum degree
 *     available in `cs` is used.
 *
 * @param[in]  max_order
 *     Maximum spherical-harmonic order to use. If negative, the maximum order
 *     available in `cs` is used. Must satisfy `max_order <= max_degree`.
 *
 * @param[in]  Re
 *     Reference radius of the spherical-harmonic model, in metres. If negative,
 *     `cs.Re()` is used.
 *
 * @param[in]  GM
 *     Gravitational parameter of the model, in m^3/s^2. If negative, `cs.GM()`
 *     is used.
 *
 * @param[in,out] M
 *     Optional scratch matrix for cosine/exterior basis terms. If supplied, it
 *     must have at least `(max_degree + 1)` rows and `(max_order + 1)` columns.
 *
 * @param[in,out] W
 *     Optional scratch matrix for sine/exterior basis terms. If supplied, it
 *     must have at least `(max_degree + 1)` rows and `(max_order + 1)` columns.
 *
 * @return
 *     `0` on success; non-zero on invalid input, insufficient scratch storage,
 *     or failure while computing basis functions.
 *
 * @warning
 *     The normalization convention of `cs.C(n,m)` and `cs.S(n,m)` must match
 *     the normalization used by `sh_basis_cs_exterior()`.
 */
// int sh_potential(
//     const dso::StokesCoeffs &cs, const Eigen::Vector3d &point,
//     double &potential, int max_degree = -1, int max_order = -1,
//     double Re = -1.0, double GM = -1.0,
//     dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M =
//         nullptr,
//     dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W =
//         nullptr) noexcept;

} /* namespace dso */

#endif
