/** @file
 * Computation of gravity acceleration using a geopotential model.
 */

#ifndef DSO_GEOPOTENTIAL_ACCELERATAION_HPP
#define DSO_GEOPOTENTIAL_ACCELERATAION_HPP

#include "eigen3/Eigen/Eigen"
#include "stokes_coefficients.hpp"

namespace dso {

namespace detail {
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
const NormalizedLegendreFactors &normalized_legendre_factors() noexcept;

struct PrecomputedShSqrts {
  static const int N = NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS;
  std::array<double, N> sqnp3;
  std::array<double, N> sqnp5;
  PrecomputedShSqrts() noexcept;
};
const PrecomputedShSqrts &precomputed_sh_sqrts() noexcept;

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

} /* namespace detail */

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
Eigen::Vector3d point_mass_acceleration(const Eigen::Vector3d &r,
                                        const Eigen::Vector3d &rcb,
                                        double GMcb) noexcept;

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
Eigen::Vector3d
point_mass_acceleration(const Eigen::Vector3d &r, const Eigen::Vector3d &rcb,
                        double GMcb,
                        Eigen::Matrix<double, 3, 3> &jacobian) noexcept;

/** @brief Compute gravitational acceleration and gravity-gradient tensor from
 *         normalized spherical-harmonic Stokes coefficients.
 *
 *  This routine evaluates the exterior gravitational potential of a body
 *  represented by normalized Stokes coefficients and returns:
 *
 *  - the Cartesian gravitational acceleration vector
 *    @f[
 *      \mathbf{a}(\mathbf{r}) = \nabla V(\mathbf{r}),
 *    @f]
 *  - and the Cartesian gravity-gradient tensor (the Hessian of the potential)
 *    @f[
 *      \mathbf{\Gamma}(\mathbf{r}) = \nabla \nabla V(\mathbf{r}).
 *    @f]
 *
 *  The potential is understood in the usual normalized spherical-harmonic form
 *  for the exterior field,
 *  @f[
 *    V(\mathbf{r}) =
 *    \frac{GM}{R_e}
 *    \sum_{n=0}^{N}
 *    \sum_{m=0}^{\min(n,M)}
 *    \left(
 *      \bar{C}_{nm}\,C_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *      +
 *      \bar{S}_{nm}\,S_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *    \right),
 *  @f]
 *  where:
 *  - @f$\bar{C}_{nm}, \bar{S}_{nm}@f$ are the normalized Stokes coefficients,
 *  - @f$C_{nm}(\mathbf{r}/R_e), S_{nm}(\mathbf{r}/R_e)@f$ are the corresponding
 *    exterior spherical-harmonic basis functions evaluated at the normalized
 *    position,
 *  - @f$GM@f$ is the gravitational parameter,
 *  - and @f$R_e@f$ is the reference radius.
 *
 *  The computation follows the Cunningham approach for Cartesian first and
 *  second derivatives of the spherical-harmonic potential. Accumulation of the
 *  harmonic contributions is arranged so that smaller-magnitude contributions
 *  are summed before the dominant low-degree terms, in order to reduce loss of
 *  numerical precision.
 *
 *  The returned quantities are expressed in Cartesian components:
 *
 *  @f[
 *    \mathbf{a} =
 *    \begin{bmatrix}
 *      a_x \\ a_y \\ a_z
 *    \end{bmatrix},
 *  @f]
 *
 *  @f[
 *    \mathbf{\Gamma} =
 *    \begin{bmatrix}
 *      \partial a_x/\partial x & \partial a_x/\partial y & \partial
 * a_x/\partial z \\
 *      \partial a_y/\partial x & \partial a_y/\partial y & \partial
 * a_y/\partial z \\
 *      \partial a_z/\partial x & \partial a_z/\partial y & \partial
 * a_z/\partial z
 *    \end{bmatrix}
 *    =
 *    \begin{bmatrix}
 *      \partial^2 V/\partial x^2 &
 *      \partial^2 V/\partial x \partial y &
 *      \partial^2 V/\partial x \partial z \\
 *      \partial^2 V/\partial y \partial x &
 *      \partial^2 V/\partial y^2 &
 *      \partial^2 V/\partial y \partial z \\
 *      \partial^2 V/\partial z \partial x &
 *      \partial^2 V/\partial z \partial y &
 *      \partial^2 V/\partial z^2
 *    \end{bmatrix}.
 *  @f]
 *
 *  For points exterior to the attracting masses, the tensor is symmetric and
 *  satisfies Laplace's equation,
 *  @f[
 *    \Gamma_{xx} + \Gamma_{yy} + \Gamma_{zz} = 0.
 *  @f]
 *
 *  @param[in] cs Normalized Stokes coefficients defining the gravity field.
 *
 *  @param[in] r  Geocentric Cartesian position vector of the evaluation point,
 *                expressed in an Earth-fixed frame (for example ECEF / ITRF),
 *                in metres.
 *
 *  @param[out] acc Cartesian gravitational acceleration vector
 *                  @f$\nabla V(\mathbf{r})@f$, in
 * @f$[\mathrm{m}\,\mathrm{s}^{-2}]@f$.
 *
 *  @param[out] gradient Cartesian gravity-gradient tensor
 *                       @f$\nabla\nabla V(\mathbf{r})@f$, in
 *                       @f$[\mathrm{s}^{-2}]@f$.
 *
 *  @param[in] max_degree Maximum spherical-harmonic degree of the expansion.
 *                        If negative, the maximum available degree in
 *                        @p cs is used.
 *
 *  @param[in] max_order  Maximum spherical-harmonic order of the expansion.
 *                        If negative, the maximum admissible order is derived
 *                        from @p cs and the chosen degree truncation.
 *
 *  @param[in] Re Reference radius @f$R_e@f$ in metres. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] GM Gravitational parameter @f$GM@f$ in
 *                @f$[\mathrm{m}^3\,\mathrm{s}^{-2}]@f$. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] W Optional scratch matrix used internally to store one family of
 *               spherical-harmonic basis values. If provided, it must be a
 *               lower-triangular, column-wise matrix of dimensions at least
 *               @f$(\texttt{max\_degree}+3) \times (\texttt{max\_degree}+3)@f$.
 *               If not provided, the routine allocates and frees the required
 *               temporary storage internally.
 *
 *  @param[in] M Optional scratch matrix used internally to store the companion
 *               family of spherical-harmonic basis values. If provided, it
 *               must be a lower-triangular, column-wise matrix of dimensions at
 *               least
 *               @f$(\texttt{max\_degree}+3) \times (\texttt{max\_degree}+3)@f$.
 *               If not provided, the routine allocates and frees the required
 *               temporary storage internally.
 *
 *  @return Zero on success. Any non-zero return value denotes failure.
 *
 *  @note This routine evaluates the exterior field only.
 *
 *  @warning If scratch matrices @p W and @p M are supplied by the caller, they
 *           must refer to distinct storage objects and must be large enough for
 *           the requested truncation.
 */
[[nodiscard]]
int sh2gradient(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, Eigen::Vector3d &acc,
    Eigen::Matrix<double, 3, 3> &gradient, int max_degree = -1,
    int max_order = -1, double Re = -1, double GM = -1,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W =
        nullptr,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M =
        nullptr) noexcept;

/** @brief Compute gravitational potential from normalized spherical-harmonic
 *         Stokes coefficients.
 *
 *  This routine evaluates the exterior gravitational potential generated by a
 *  set of normalized Stokes coefficients at a geocentric Cartesian position.
 *
 *  The potential is assumed in the normalized exterior spherical-harmonic form
 *  @f[
 *    V(\mathbf{r}) =
 *    \frac{GM}{R_e}
 *    \sum_{n=0}^{N}
 *    \sum_{m=0}^{\min(n,M)}
 *    \left(
 *      \bar{C}_{nm}\,C_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *      +
 *      \bar{S}_{nm}\,S_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *    \right),
 *  @f]
 *  where:
 *  - @f$\bar{C}_{nm}, \bar{S}_{nm}@f$ are the normalized Stokes coefficients,
 *  - @f$C_{nm}, S_{nm}@f$ are the corresponding exterior basis functions,
 *  - @f$GM@f$ is the gravitational parameter,
 *  - and @f$R_e@f$ is the reference radius.
 *
 *  The computation is performed using the same exterior spherical-harmonic
 *  basis functions employed in the Cunningham-style derivative routines, but
 *  here only the scalar potential itself is accumulated. Harmonic
 *  contributions are summed from smaller to larger magnitude in order to
 *  reduce loss of numerical precision.
 *
 *  @param[in] cs Normalized Stokes coefficients defining the gravity field.
 *
 *  @param[in] r Geocentric Cartesian position vector of the evaluation point,
 *               expressed in an Earth-fixed frame (for example ECEF / ITRF),
 *               in metres.
 *
 *  @param[out] U Gravitational potential at the evaluation point, in
 *                @f$[\mathrm{m}^2\,\mathrm{s}^{-2}]@f$.
 *
 *  @param[in] max_degree Maximum spherical-harmonic degree of the expansion.
 *                        If negative, the maximum available degree in
 *                        @p cs is used.
 *
 *  @param[in] max_order Maximum spherical-harmonic order of the expansion.
 *                       If negative, the maximum admissible order is derived
 *                       from @p cs and the chosen degree truncation.
 *
 *  @param[in] Re Reference radius @f$R_e@f$ in metres. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] GM Gravitational parameter @f$GM@f$ in
 *                @f$[\mathrm{m}^3\,\mathrm{s}^{-2}]@f$. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] W Optional scratch matrix used internally to store one family of
 *               spherical-harmonic basis values. If provided, it must be a
 *               lower-triangular, column-wise matrix of dimensions at least
 *               @f$(\texttt{max\_degree}+1) \times (\texttt{max\_degree}+1)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @param[in] M Optional scratch matrix used internally to store the companion
 *               family of spherical-harmonic basis values. If provided, it
 *               must be a lower-triangular, column-wise matrix of dimensions at
 *               least
 *               @f$(\texttt{max\_degree}+1) \times (\texttt{max\_degree}+1)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @return Zero on success. Any non-zero return value denotes failure.
 *
 *  @note This routine evaluates the exterior field only.
 *
 *  @warning If scratch matrices @p W and @p M are supplied by the caller, they
 *           must refer to distinct storage objects and must be large enough for
 *           the requested truncation.
 */
[[nodiscard]]
int sh2potential(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, double &U,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept;

/** @brief Compute gravitational acceleration from normalized spherical-harmonic
 *         Stokes coefficients.
 *
 *  This routine evaluates the exterior gravitational potential defined by a
 *  set of normalized Stokes coefficients and returns its Cartesian gradient,
 *  i.e. the gravitational acceleration vector
 *  @f[
 *    \mathbf{g}(\mathbf{r}) = \nabla V(\mathbf{r}).
 *  @f]
 *
 *  The potential is assumed in the normalized exterior spherical-harmonic form
 *  @f[
 *    V(\mathbf{r}) =
 *    \frac{GM}{R_e}
 *    \sum_{n=0}^{N}
 *    \sum_{m=0}^{\min(n,M)}
 *    \left(
 *      \bar{C}_{nm}\,C_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *      +
 *      \bar{S}_{nm}\,S_{nm}\!\left(\frac{\mathbf{r}}{R_e}\right)
 *    \right),
 *  @f]
 *  where:
 *  - @f$\bar{C}_{nm}, \bar{S}_{nm}@f$ are the normalized Stokes coefficients,
 *  - @f$C_{nm}, S_{nm}@f$ are the corresponding exterior basis functions,
 *  - @f$GM@f$ is the gravitational parameter,
 *  - and @f$R_e@f$ is the reference radius.
 *
 *  The Cartesian acceleration is computed using the Cunningham formulation for
 *  first derivatives of the spherical-harmonic potential. Harmonic
 *  contributions are accumulated from smaller to larger magnitude in order to
 *  reduce loss of numerical precision.
 *
 *  The returned vector is
 *  @f[
 *    \mathbf{g} =
 *    \begin{bmatrix}
 *      g_x \\ g_y \\ g_z
 *    \end{bmatrix}
 *    =
 *    \begin{bmatrix}
 *      \partial V/\partial x \\
 *      \partial V/\partial y \\
 *      \partial V/\partial z
 *    \end{bmatrix},
 *  @f]
 *  expressed in Cartesian components.
 *
 *  @param[in] cs Normalized Stokes coefficients defining the gravity field.
 *
 *  @param[in] r Geocentric Cartesian position vector of the evaluation point,
 *               expressed in an Earth-fixed frame (for example ECEF / ITRF),
 *               in metres.
 *
 *  @param[out] gravity Cartesian gravitational acceleration vector
 *                      @f$\nabla V(\mathbf{r})@f$, in
 *                      @f$[\mathrm{m}\,\mathrm{s}^{-2}]@f$.
 *
 *  @param[in] max_degree Maximum spherical-harmonic degree of the expansion.
 *                        If negative, the maximum available degree in
 *                        @p cs is used.
 *
 *  @param[in] max_order Maximum spherical-harmonic order of the expansion.
 *                       If negative, the maximum admissible order is derived
 *                       from @p cs and the chosen degree truncation.
 *
 *  @param[in] Re Reference radius @f$R_e@f$ in metres. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] GM Gravitational parameter @f$GM@f$ in
 *                @f$[\mathrm{m}^3\,\mathrm{s}^{-2}]@f$. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] W Optional scratch matrix used internally to store one family of
 *               spherical-harmonic basis values. If provided, it must be a
 *               lower-triangular, column-wise matrix of dimensions at least
 *               @f$(\texttt{max\_degree}+2) \times (\texttt{max\_degree}+2)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @param[in] M Optional scratch matrix used internally to store the companion
 *               family of spherical-harmonic basis values. If provided, it
 *               must be a lower-triangular, column-wise matrix of dimensions at
 *               least
 *               @f$(\texttt{max\_degree}+2) \times (\texttt{max\_degree}+2)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @return Zero on success. Any non-zero return value denotes failure.
 *
 *  @note This routine evaluates the exterior field only.
 *
 *  @warning If scratch matrices @p W and @p M are supplied by the caller, they
 *           must refer to distinct storage objects and must be large enough for
 *           the requested truncation.
 */
[[nodiscard]]
int sh2gravity(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r,
    Eigen::Vector3d &gravity, int max_degree, int max_order, double Re,
    double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept;

/** @brief Compute elastic surface displacement due to a load field represented
 *         by normalized spherical-harmonic Stokes coefficients.
 *
 *  This routine evaluates, at a geocentric Cartesian ECEF position, the elastic
 *  displacement induced by a load field represented by normalized Stokes
 *  coefficients.
 *
 *  The load field is first used to compute:
 *
 *  - the scalar perturbing potential \f$ V(\mathbf{r}) \f$
 *  - and its Cartesian gradient \f$ \nabla V(\mathbf{r}) \f$
 *
 *  by means of a Cunningham-style spherical-harmonic evaluation of the first
 *  derivatives of the exterior potential.
 *
 *  The deformation is then assembled degree by degree using the load Love
 *  numbers:
 *  @f[
 *    \mathbf{d} =
 *    \sum_{n=0}^{N}
 *    \left[
 *      \frac{h_n}{g}\,V_n\,\hat{\mathbf{u}}
 *      +
 *      \frac{l_n}{g}
 *      \left(
 *        \nabla V_n - (\nabla V_n \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}
 *      \right)
 *    \right],
 *  @f]
 *  where:
 *  - @f$h_n@f$ are the vertical load Love numbers,
 *  - @f$l_n@f$ are the horizontal load Love numbers,
 *  - @f$g@f$ is the local background gravity magnitude at the site,
 *  - and @f$\hat{\mathbf{u}} = \mathbf{r}/\|\mathbf{r}\|@f$ is the outward
 *    radial unit vector.
 *
 *  @param[in] cs Normalized Stokes coefficients describing the load field.
 *
 *  @param[in] r Geocentric Cartesian position vector of the evaluation point,
 *               expressed in an Earth-fixed frame, in metres.
 *
 *  @param[out] dr Elastic deformation vector in Cartesian components, in
 *                 metres.
 *
 *  @param[in] gravity Local background gravity magnitude at the site, in
 *                     \f$[\mathrm{m}\,\mathrm{s}^{-2}]\f$.
 *                     This quantity is independent of the load field supplied
 *                     through @p cs.
 *
 *  @param[out] potential Scalar perturbing potential at the evaluation point,
 *                        in \f$[\mathrm{m}^2\,\mathrm{s}^{-2}]\f$.
 *
 *  @param[out] potential_grad Cartesian gradient of the perturbing potential,
 *                             i.e. \f$\nabla V(\mathbf{r})\f$, in
 *                             \f$[\mathrm{m}\,\mathrm{s}^{-2}]\f$.
 *
 *  @param[in] max_degree Maximum spherical-harmonic degree. If negative, the
 *                        maximum available degree in @p cs is used.
 *
 *  @param[in] max_order Maximum spherical-harmonic order. If negative, the
 *                       maximum admissible order is derived from @p cs and the
 *                       chosen degree truncation.
 *
 *  @param[in] Re Reference radius \f$R_e\f$ in metres. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] GM Gravitational parameter \f$GM\f$ in
 *                \f$[\mathrm{m}^3\,\mathrm{s}^{-2}]\f$. If negative, the value
 *                stored in @p cs is used.
 *
 *  @param[in] W Optional scratch matrix used internally to store one family of
 *               spherical-harmonic basis values. If provided, it must be a
 *               lower-triangular, column-wise matrix of dimensions at least
 *               @f$(\texttt{max\_degree}+2) \times (\texttt{max\_degree}+2)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @param[in] M Optional scratch matrix used internally to store the companion
 *               family of spherical-harmonic basis values. If provided, it
 *               must be a lower-triangular, column-wise matrix of dimensions at
 *               least
 *               @f$(\texttt{max\_degree}+2) \times (\texttt{max\_degree}+2)@f$.
 *               If not provided, temporary storage is allocated and released
 *               internally.
 *
 *  @return Zero on success. Any non-zero return value denotes failure.
 *
 *  @note This routine evaluates the exterior field only.
 *
 *  @note The scalar parameter @p gravity is the background local gravity
 *        magnitude used in the deformation formula. It is not the norm of the
 *        perturbing load-induced gradient returned in @p potential_grad.
 *
 *  @warning If scratch matrices @p W and @p M are supplied by the caller, they
 *           must refer to distinct storage objects and must be large enough for
 *           the requested truncation.
 */
[[nodiscard]]
int sh2deformation(
    const dso::StokesCoeffs &cs, const Eigen::Vector3d &r, Eigen::Vector3d &dr,
    double gravity, double &potential, Eigen::Vector3d &potential_grad,
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

} /* namespace dso */

#endif
