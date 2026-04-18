/** @file coeff_matrix_2d_documented.hpp
 *  @brief Compact 2D coefficient matrix with compile-time storage policy.
 *
 *  This header defines @ref dso::CoeffMatrix2D, a matrix-like container whose
 *  physical storage layout is chosen at compile time through the template
 *  parameter @ref dso::MatrixStorageType.
 *
 *  The design goal of this class is:
 *  - to store compact coefficient tables efficiently,
 *  - to expose the logical 2D interface `A(i,j)`,
 *  - and to support a *small*, performance-oriented set of arithmetic patterns
 *    needed by the surrounding code.
 *
 *  In particular, this header supports:
 *  - direct element access,
 *  - storage-aware resize / copy-resize,
 *  - reduced (top-left) views through lightweight proxy objects,
 *  - lazy expression templates for:
 *      - matrix scaling, `s * A`
 *      - matrix addition, `A + B`
 *  - optimized in-place kernels for selected storage/pattern combinations
 *    (currently AVX2-specialized kernels for
 *    @ref dso::MatrixStorageType::LwTriangularColWise).
 *
 *  ## Why is this class template parameterized by storage?
 *
 *  The key idea is that the matrix *shape semantics* and the matrix *memory
 *  layout* are separated:
 *
 *  - @ref dso::StorageImplementation tells us how a logical `(row,col)` maps
 *    into a flat contiguous buffer.
 *  - @ref dso::CoeffMatrix2D owns that flat contiguous buffer.
 *
 *  This yields a few important properties:
 *  - no virtual dispatch,
 *  - layout-specific indexing resolved at compile time,
 *  - efficient contiguous flat loops over the underlying storage when possible.
 *
 *  ## Arithmetic philosophy
 *
 *  This class is **not** a general-purpose linear-algebra matrix.
 *  It supports only the operations that are useful for the coefficient-storage
 *  use case:
 *
 *  - `A += rhs`
 *  - `s * A`
 *  - `A + B`
 *  - `A.reduced_view(r,c)`
 *  - specialized in-place "AXPY"-like kernels for selected layouts
 *
 *  These operations are intentionally narrow. The purpose is to keep the class
 *  efficient, predictable, and easy for the compiler to optimize.
 *
 *  ## Storage and ownership
 *
 *  A @ref dso::CoeffMatrix2D instance owns:
 *  - a storage-policy object (`m_storage`)
 *  - an aligned or unaligned heap buffer (`m_data`)
 *  - the capacity of the currently allocated memory arena (`_capacity`)
 *
 *  The proxies defined in this header do **not** allocate. They either:
 *  - borrow existing expressions by reference, or
 *  - own a small temporary expression object by value
 *
 *  depending on the value category of the originating expression.
 */

#ifndef __COMPACT_2D_SIMPLE_MATRIX_HPP__
#define __COMPACT_2D_SIMPLE_MATRIX_HPP__

#include "coeff_matrix_storage.hpp"
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#ifdef DEBUG
#include <cassert>
#include <cstdio>
#endif
#include <cstddef>
#include <cstdint>
#ifdef DSO_SIMD
#include <immintrin.h>
#endif

/** @def DSO_RESTRICT
 *  @brief Compiler-specific spelling of a C/C++ "restrict"-like qualifier.
 *
 *  This macro is used only for low-level pointer kernels. It communicates to
 *  the compiler that the pointed-to memory ranges are assumed not to alias.
 *
 *  @note
 *  This is a performance contract. Passing aliased buffers to functions using
 *  `DSO_RESTRICT` is outside the intended usage contract and may lead to wrong
 *  code generation.
 */
#if defined(__GNUC__) || defined(__clang__)
#define DSO_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define DSO_RESTRICT __restrict
#else
#define DSO_RESTRICT
#endif

/** @namespace detail
 *  @brief Internal helpers for expression-template dispatch and low-level
 * kernels.
 *
 *  The entities in this namespace are implementation details. They are used by
 *  @ref dso::CoeffMatrix2D but are not meant to be part of the public API.
 *
 *  The namespace currently contains:
 *  - tiny assignment helpers for `reduce_copy`
 *  - expression-detection traits
 *  - optional C++20 concept checks for storage construction
 *  - low-level scalar/SIMD kernels for selected in-place operations
 */
namespace detail {

/** @brief Assign `rhs` to `lhs`.
 *  @param[in,out] lhs Left-hand side value to overwrite.
 *  @param[in] rhs New value.
 *  @return Reference to `lhs` after assignment.
 *
 *  This helper exists so that @ref op can select assignment behavior at compile
 *  time without duplicating the surrounding traversal logic.
 */
inline double &op_equal(double &lhs, double rhs) noexcept { return lhs = rhs; }

/** @brief Add `rhs` to `lhs`.
 *  @param[in,out] lhs Left-hand side value to update.
 *  @param[in] rhs Value to add.
 *  @return Reference to `lhs` after the update.
 *
 *  This helper is used by @ref op for the `EqAdd` case.
 */
inline double &op_eqadd(double &lhs, double rhs) noexcept { return lhs += rhs; }

/** @brief Selects how an expression is reduced/copied into a matrix.
 *
 *  The same traversal logic can be used either to:
 *  - assign into a target matrix (`Equal`), or
 *  - accumulate into a target matrix (`EqAdd`)
 *
 *  @see op
 *  @see dso::CoeffMatrix2D::reduce_copy
 */
enum class ReductionAssignmentOperator { Equal, EqAdd };

/** @brief Apply the compile-time selected assignment operator.
 *  @tparam Op Either `Equal` or `EqAdd`.
 *  @param[in,out] lhs Destination coefficient.
 *  @param[in] rhs Source coefficient.
 *  @return Reference to `lhs` after the operation.
 *
 *  This is a tiny abstraction that allows the same traversal loop to serve both
 *  "copy from expression" and "accumulate expression" code paths.
 */
template <ReductionAssignmentOperator Op>
inline double &op(double &lhs, double rhs) noexcept {
  if constexpr (Op == ReductionAssignmentOperator::Equal)
    return op_equal(lhs, rhs);
  else
    return op_eqadd(lhs, rhs);
}

#if __cplusplus >= 202002L /* concepts only available in C++20 */
/** @brief Helper concept: true if `T{r}` is a valid construction.
 *
 *  This is used to provide more readable compile-time diagnostics in
 *  @ref dso::CoeffMatrix2D::make_storage_impl.
 *
 *  It matters because some storage policies are square-only and therefore must
 *  be constructible from a single integer dimension.
 */
template <class T>
concept c20xSquareConstructible = requires(int r) { T{r}; };

/** @brief Helper concept: true if `T{r,c}` is a valid construction.
 *
 *  This is used to provide more readable compile-time diagnostics in
 *  @ref dso::CoeffMatrix2D::make_storage_impl.
 *
 *  It matters because rectangular storage policies need both row and column
 *  dimensions.
 */
template <class T>
concept c20xRectConstructible = requires(int r, int c) { T{r, c}; };
#endif

/** @brief Trait detecting whether a type behaves like a matrix expression.
 *
 *  By design, expression-template nodes in this header are intentionally
 *  lightweight and do not share a common base class. Instead, "expression-ness"
 *  is detected structurally (duck-typing style).
 *
 *  A type is considered a valid expression if it provides:
 *  - `rows()`
 *  - `cols()`
 *  - `operator()(int,int)`
 *
 *  This allows:
 *  - real matrices (`CoeffMatrix2D`)
 *  - reduced-view proxies
 *  - scaled proxies
 *  - sum proxies
 *
 *  to participate uniformly in generic expression operators.
 */
template <class T, class = void> struct _is_expr : std::false_type {};

/** @brief Specialization marking a type as an expression when it provides the
 *         required query and indexing interface.
 */
template <class T>
struct _is_expr<T, std::void_t<decltype(std::declval<const T &>().rows()),
                               decltype(std::declval<const T &>().cols()),
                               decltype(std::declval<const T &>()(0, 0))>>
    : std::true_type {};

/** @brief Convenience variable template for @ref _is_expr.
 *  @tparam T Candidate type.
 */
template <class T>
static constexpr bool _is_expr_v =
    _is_expr<std::remove_cv_t<std::remove_reference_t<T>>>::value;

/** @brief Scalar AXPY kernel: `a[i] += s * b[i]`.
 *
 *  This is the fallback implementation used when SIMD is not enabled or when
 *  no specialized SIMD path is provided for a given pattern.
 *
 *  @param[in,out] a Destination contiguous array.
 *  @param[in] b Source contiguous array.
 *  @param[in] s Scalar multiplier.
 *  @param[in] n Number of elements.
 *
 *  @pre `a` and `b` point to valid storage of at least `n` doubles.
 *  @pre The storage ranges are assumed not to alias, due to `DSO_RESTRICT`.
 *
 *  @note
 *  This function is intentionally tiny and straightforward. It also serves as
 *  the semantic reference for the SIMD-accelerated version.
 */
inline void axpy_scalar(double *DSO_RESTRICT a, const double *DSO_RESTRICT b,
                        double s, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    a[i] += s * b[i];
  }
}

/** @brief AXPY kernel specialized for contiguous lower-triangular column-wise
 *         storage.
 *
 *  This operation performs:
 *  @code
 *  a[i] += s * b[i],  i = 0 .. n-1
 *  @endcode
 *
 *  The function is written specifically for the case where the logical matrices
 *  are @ref dso::MatrixStorageType::LwTriangularColWise and therefore their
 *  compact representation is a single contiguous 1D buffer of length `n`.
 *
 *  @param[in,out] a Destination buffer.
 *  @param[in] b Source buffer.
 *  @param[in] s Scalar multiplier.
 *  @param[in] n Number of stored coefficients.
 *
 *  @pre The buffers are contiguous and hold at least `n` doubles.
 *  @pre The buffers are assumed not to alias.
 *
 *  @remark
 *  In non-SIMD builds this calls @ref axpy_scalar.
 *
 *  @remark
 *  In SIMD builds this uses AVX2 instructions operating on 4 doubles at a time,
 *  with a 2x unrolled main loop (8 doubles per iteration), followed by a
 *  secondary 4-wide vector loop and then a scalar tail.
 *
 *  @warning
 *  The SIMD implementation uses aligned AVX loads/stores (`_mm256_load_pd`,
 *  `_mm256_store_pd`), so the data pointer is expected to satisfy the alignment
 *  guarantee imposed by @ref dso::CoeffMatrix2D::allocate_buffer when
 *  `DSO_SIMD` is enabled.
 */
inline void axpy_lwtri_colwise(double *DSO_RESTRICT a,
                               const double *DSO_RESTRICT b, double s,
                               std::size_t n) noexcept {
#ifndef DSO_SIMD
  axpy_scalar(a, b, s, n);
#else
  /* Broadcast the scalar factor s to all 4 lanes of a 256-bit AVX register.
   * Each __m256d holds 4 doubles. We will reuse this register in every vector
   * iteration, so the broadcast is done once up front. */
  const __m256d vs = _mm256_set1_pd(s);

  /* Flat index over the contiguous storage buffers a[] and b[].
   * For LwTriangularColWise the full matrix content is stored contiguously,
   * so this kernel intentionally ignores any 2D indexing and operates only
   * on raw linear memory. */
  std::size_t i = 0;

  /* Main vector loop, unrolled by 2 AVX vectors = 8 doubles per iteration.
   *
   * Why unroll?
   * - It reduces loop overhead.
   * - It gives the compiler / CPU a bit more instruction-level parallelism.
   * - It often improves throughput on modern x86 cores.
   *
   * The loop condition i + 8 <= n ensures that the 8 elements accessed here
   * are always in-bounds. */
  for (; i + 8 <= n; i += 8) {
    /* Load 4 consecutive doubles from a[i .. i+3].
     * Because we plan to use aligned allocation for matrix buffers, _load_pd
     * is the intended aligned load here. */
    __m256d va0 = _mm256_load_pd(a + i);

    /* Load 4 consecutive doubles from b[i .. i+3]. */
    __m256d vb0 = _mm256_load_pd(b + i);

    /* Load the next 4 doubles from a[i+4 .. i+7]. */
    __m256d va1 = _mm256_load_pd(a + i + 4);

    /* Load the next 4 doubles from b[i+4 .. i+7]. */
    __m256d vb1 = _mm256_load_pd(b + i + 4);

#if defined(__FMA__)
    /* Fused multiply-add:
     *   va0 = vb0 * vs + va0
     *   va1 = vb1 * vs + va1
     *
     * This computes exactly the desired daxpy-style update:
     *   a[k] += s * b[k]
     *
     * Using FMA can reduce instruction count and can improve both throughput
     * and numerical behavior slightly because multiplication and addition are
     * fused into one instruction. */
    va0 = _mm256_fmadd_pd(vb0, vs, va0);
    va1 = _mm256_fmadd_pd(vb1, vs, va1);
#else
    /* Non-FMA fallback:
     * First compute the vector product vb*vs, then add it to va.
     *
     * This is still fully vectorized AVX2 code, just using separate multiply
     * and add instructions instead of a fused instruction. */
    va0 = _mm256_add_pd(va0, _mm256_mul_pd(vb0, vs));
    va1 = _mm256_add_pd(va1, _mm256_mul_pd(vb1, vs));
#endif

    /* Store the updated first 4 doubles back into a[i .. i+3]. */
    _mm256_store_pd(a + i, va0);

    /* Store the updated next 4 doubles back into a[i+4 .. i+7]. */
    _mm256_store_pd(a + i + 4, va1);
  }

  /* Secondary vector loop handling one AVX vector = 4 doubles at a time.
   *
   * This loop runs after the 8-wide unrolled loop and processes any remaining
   * full 4-double chunk. The loop condition i + 4 <= n guarantees in-bounds
   * access for a single vector. */
  for (; i + 4 <= n; i += 4) {
    /* Load 4 doubles from a. */
    __m256d va = _mm256_load_pd(a + i);

    /* Load the matching 4 doubles from b. */
    __m256d vb = _mm256_load_pd(b + i);

#if defined(__FMA__)
    /* Vector fused multiply-add for one 4-double chunk. */
    va = _mm256_fmadd_pd(vb, vs, va);
#else
    /* Vector multiply + add fallback for one 4-double chunk. */
    va = _mm256_add_pd(va, _mm256_mul_pd(vb, vs));
#endif

    /* Store the updated vector back to a. */
    _mm256_store_pd(a + i, va);
  }

  /* Scalar cleanup loop.
   *
   * If n is not divisible by 8 or 4, there can be 1, 2, or 3 elements left.
   * Those are handled here with ordinary scalar arithmetic.
   *
   * This keeps the vector loops simple and avoids any masked-tail logic. */
  for (; i < n; ++i) {
    /* Final scalar daxpy update for the leftover elements. */
    a[i] += s * b[i];
  }
#endif
}

/** @brief Two-source AXPY-like kernel specialized for contiguous
 * lower-triangular column-wise storage.
 *
 *  This operation performs:
 *  @code
 *  a[i] += s1 * b1[i] + s2 * b2[i],  i = 0 .. n-1
 *  @endcode
 *
 *  This kernel is the low-level engine behind
 *  @ref dso::CoeffMatrix2D::axpy2_inplace for
 *  @ref dso::MatrixStorageType::LwTriangularColWise.
 *
 *  @param[in,out] a Destination buffer.
 *  @param[in] b1 First source buffer.
 *  @param[in] b2 Second source buffer.
 *  @param[in] s1 Scalar multiplier for `b1`.
 *  @param[in] s2 Scalar multiplier for `b2`.
 *  @param[in] n Number of stored coefficients.
 *
 *  @pre All buffers are contiguous and hold at least `n` doubles.
 *  @pre The buffers are assumed not to alias.
 *
 *  @warning
 *  Aliasing patterns such as `a == b1`, `a == b2`, or `b1 == b2` are outside
 *  the intended usage contract because the pointers carry `DSO_RESTRICT`.
 */
inline void axpy2_lwtri_colwise(double *DSO_RESTRICT a,
                                const double *DSO_RESTRICT b1,
                                const double *DSO_RESTRICT b2, double s1,
                                double s2, std::size_t n) noexcept {
#ifndef DSO_SIMD
  for (std::size_t i = 0; i < n; ++i) {
    a[i] += s1 * b1[i] + s2 * b2[i];
  }
#else
  /* Broadcast scalar s1 to all lanes of one AVX register. */
  const __m256d vs1 = _mm256_set1_pd(s1);

  /* Broadcast scalar s2 to all lanes of one AVX register. */
  const __m256d vs2 = _mm256_set1_pd(s2);

  /* Flat index over the contiguous storage arrays. */
  std::size_t i = 0;

  /* Main vector loop, unrolled by 2 AVX vectors = 8 doubles per iteration. */
  for (; i + 8 <= n; i += 8) {
    /* Load 4 doubles from A, B1, B2. */
    __m256d va0 = _mm256_load_pd(a + i);
    __m256d vb10 = _mm256_load_pd(b1 + i);
    __m256d vb20 = _mm256_load_pd(b2 + i);

    /* Load next 4 doubles from A, B1, B2. */
    __m256d va1 = _mm256_load_pd(a + i + 4);
    __m256d vb11 = _mm256_load_pd(b1 + i + 4);
    __m256d vb21 = _mm256_load_pd(b2 + i + 4);

#if defined(__FMA__)
    /* First fused update:
     *   va = s1 * b1 + va
     */
    va0 = _mm256_fmadd_pd(vb10, vs1, va0);
    va1 = _mm256_fmadd_pd(vb11, vs1, va1);

    /* Second fused update:
     *   va = s2 * b2 + va
     *
     * After the two FMAs, we have:
     *   va = a + s1*b1 + s2*b2
     */
    va0 = _mm256_fmadd_pd(vb20, vs2, va0);
    va1 = _mm256_fmadd_pd(vb21, vs2, va1);
#else
    /* Non-FMA fallback:
     *   va = va + s1*b1 + s2*b2
     */
    va0 = _mm256_add_pd(va0, _mm256_mul_pd(vb10, vs1));
    va1 = _mm256_add_pd(va1, _mm256_mul_pd(vb11, vs1));

    va0 = _mm256_add_pd(va0, _mm256_mul_pd(vb20, vs2));
    va1 = _mm256_add_pd(va1, _mm256_mul_pd(vb21, vs2));
#endif

    /* Store updated values back to A. */
    _mm256_store_pd(a + i, va0);
    _mm256_store_pd(a + i + 4, va1);
  }

  /* Secondary vector loop handling one AVX vector = 4 doubles. */
  for (; i + 4 <= n; i += 4) {
    __m256d va = _mm256_load_pd(a + i);
    __m256d vb1 = _mm256_load_pd(b1 + i);
    __m256d vb2 = _mm256_load_pd(b2 + i);

#if defined(__FMA__)
    va = _mm256_fmadd_pd(vb1, vs1, va);
    va = _mm256_fmadd_pd(vb2, vs2, va);
#else
    va = _mm256_add_pd(va, _mm256_mul_pd(vb1, vs1));
    va = _mm256_add_pd(va, _mm256_mul_pd(vb2, vs2));
#endif

    _mm256_store_pd(a + i, va);
  }

  /* Scalar cleanup for the remaining 0..3 elements. */
  for (; i < n; ++i) {
    a[i] += s1 * b1[i] + s2 * b2[i];
  }
#endif
}
} /* namespace detail */

namespace dso {

/** @brief Compact 2D coefficient matrix with compile-time storage policy.
 *
 *  @tparam S Storage policy, one of the values of @ref MatrixStorageType.
 *
 *  This class owns a flat `double` buffer and presents it as a logical 2D
 *  matrix. The actual mapping from `(row,col)` to 1D storage is delegated to
 *  @ref StorageImplementation.
 *
 *  ## Main responsibilities
 *
 *  - own and manage the flat memory buffer,
 *  - expose logical 2D indexing through `operator()(i,j)`,
 *  - provide row/column slice access when the storage policy allows it,
 *  - support a small set of arithmetic patterns via lightweight expression
 *    templates,
 *  - provide selected hand-optimized in-place kernels for hot operations.
 *
 *  ## What this class is not
 *
 *  This class is not meant to be a full linear-algebra matrix package.
 *  It intentionally supports only a narrow and performance-driven subset of
 *  operations useful for coefficient-table manipulation.
 *
 *  ## Expression templates in this class
 *
 *  The nested helper types:
 *  - @ref _ReducedViewProxy
 *  - @ref _ScaledProxy
 *  - @ref _SumProxy
 *
 *  allow expressions such as:
 *
 *  @code
 *  auto rv = A.reduced_view(10, 5);
 *  auto sp = 2.0 * A;
 *  auto sum = A + B;
 *  auto expr = 0.5 * A + 2.0 * B;
 *  auto cropped = (A + B).reduced_view(20, 20);
 *  @endcode
 *
 *  without materializing temporary full matrices unless and until the result is
 *  explicitly copied into a @ref CoeffMatrix2D.
 */
template <MatrixStorageType S> class CoeffMatrix2D {
private:
  /** @brief Alias for the compile-time storage policy. */
  using Storage = StorageImplementation<S>;

  Storage m_storage;        /** storage type; dictates indexing */
  double *m_data{nullptr};  /** the actual data */
  std::size_t _capacity{0}; /** number of doubles in allocated memory arena */

  /** @brief This class always owns a single flat contiguous buffer. */
  static constexpr const int hasContiguousMem = true;

  /** @brief Byte alignment used for SIMD-enabled allocations.
   *
   *  A 64-byte alignment is conservative and friendly to modern cache-line
   *  sizes while also satisfying 32-byte AVX alignment requirements.
   */
  static constexpr std::size_t simd_alignment = 64;

  /** @brief Allocate storage for `n` doubles.
   *  @param[in] n Number of doubles.
   *  @return Pointer to newly allocated storage or `nullptr` if `n == 0`.
   *
   *  In SIMD builds, aligned allocation is used so that the low-level AVX
   *  kernels can legally use aligned loads/stores.
   *
   *  @warning
   *  The returned storage must be released through
   *  @ref deallocate_buffer, not through a mismatched delete.
   */
  static double *allocate_buffer(const std::size_t n) {
    if (n == 0)
      return nullptr;
#ifdef DSO_SIMD
    return static_cast<double *>(
        ::operator new[](n * sizeof(double), std::align_val_t{simd_alignment}));
#else
    return new double[n];
#endif
  }

  /** @brief Release storage acquired by @ref allocate_buffer.
   *  @param[in] ptr Pointer previously returned by @ref allocate_buffer.
   */
  static void deallocate_buffer(double *ptr) noexcept {
    if (!ptr)
      return;
#ifdef DSO_SIMD
    ::operator delete[](ptr, std::align_val_t{simd_alignment});
#else
    delete[] ptr;
#endif
  }

  /** @brief Access one coefficient in flat contiguous storage.
   *
   *  This helper intentionally bypasses logical `(row,col)` indexing.
   *  It is used only in code paths where contiguous storage access is known
   *  to be valid and beneficial.
   *
   *  @param[in] i Flat storage index.
   *  @return Stored value at offset `i`.
   *
   *  @warning
   *  This is an implementation helper. Callers must already know that `i`
   *  refers to a valid flat offset in the underlying storage.
   */
  double data(int i) const noexcept { return m_data[i]; }

  /** @brief Return a pointer to the beginning of a row/column slice together
   *         with its stored length.
   *
   *  The meaning of the slice depends on the storage policy:
   *  - row-major storages: the `i`-th row
   *  - column-major storages: the `i`-th column
   *
   *  @param[in] i Zero-based row/column selector.
   *  @param[out] num_elements Number of stored coefficients in this slice.
   *  @return Const pointer to the first stored coefficient of the slice.
   *
   *  @see StorageImplementation::slice(int, int&)
   */
  const double *slice(int i, int &num_elements) const noexcept {
    return m_data + m_storage.slice(i, num_elements);
  }

  /** @brief Mutable variant of @ref slice(int, int&) const.
   *  @param[in] i Zero-based row/column selector.
   *  @param[out] num_elements Number of stored coefficients in this slice.
   *  @return Mutable pointer to the first stored coefficient of the slice.
   */
  double *slice(int i, int &num_elements) noexcept {
    return m_data + m_storage.slice(i, num_elements);
  }

  /** @brief Construct a storage policy object for the requested dimensions.
   *
   *  This helper centralizes the logic:
   *  - square storage policies are built from a single dimension
   *  - rectangular storage policies are built from `(rows, cols)`
   *
   *  @param[in] rows Requested logical row count.
   *  @param[in] cols Requested logical column count.
   *  @return A storage object of type `Storage`.
   *
   *  @remark
   *  Keeping this logic in one place avoids repeating square-vs-rectangular
   *  construction rules throughout constructors and resize functions.
   */
  static Storage make_storage_impl(int rows, int cols) {
    if constexpr (Storage::isSquare) {
#if __cplusplus >= 202002L
      static_assert(detail::c20xSquareConstructible<Storage>,
                    "Square storage must be constructible as Storage(int)");
#endif
#ifdef DEBUG
      assert(rows == cols);
#endif
      (void)cols;
      return Storage(rows);
    } else {
#if __cplusplus >= 202002L
      static_assert(detail::c20xRectConstructible<Storage>,
                    "Rect storage must be constructible as Storage(int,int)");
#endif
      return Storage(rows, cols);
    }
  }

  /** @brief Resize the matrix without preserving previous values.
   *
   *  This is the low-level "destructive resize" helper behind the public
   *  `resize(...)` overloads.
   *
   *  Existing values are considered irrelevant after this call. The storage
   *  buffer may be reallocated if the requested number of elements exceeds the
   *  current capacity.
   *
   *  @param[in] rows New logical row count.
   *  @param[in] cols New logical column count.
   *
   *  @warning
   *  This function does *not* preserve the prior contents of the matrix.
   *  If you need shape changes while keeping the overlapping data region,
   *  use @ref cresize_impl instead.
   */
  void resize_impl(int rows, int cols) noexcept {
    /* new number of elements (after resize) */
    const auto nele = make_storage_impl(rows, cols).num_elements();
    /* do we need to re-allocate ? */
    if (nele > _capacity) {
      if (m_data)
        deallocate_buffer(m_data);
      m_data = allocate_buffer(nele);
      _capacity = nele;
    }
    /* set correct storage */
    m_storage = make_storage_impl(rows, cols);
  }

  /** @brief Resize the matrix while preserving the overlap with the old shape.
   *
   *  This helper implements the semantics of the public `cresize(...)` API:
   *  values that belong to the overlap between old and new shapes are copied
   *  into the new storage arena.
   *
   *  @param[in] rows New logical row count.
   *  @param[in] cols New logical column count.
   *
   *  @note
   *  The copy is performed slice-by-slice. Each source slice and target slice
   *  can have different logical lengths depending on the storage policy.
   */
  void cresize_impl(int rows, int cols) {
    /* no-op */
    if (rows == this->rows() && cols == this->cols()) {
      return;
    }

    Storage new_storage = make_storage_impl(rows, cols);
    const auto num_elements = new_storage.num_elements();

    double *ptr = allocate_buffer(num_elements);
    if (m_data) {
      int num_doubles_src;
      int num_doubles_trg;
      /* copy data (from m_data to ptr) */
      for (int s = 0;
           s < std::min(new_storage.num_slices(), m_storage.num_slices());
           s++) {
        const double *__restrict__ psrc = this->slice(s, num_doubles_src);
        double *__restrict__ ptrg = ptr + new_storage.slice(s, num_doubles_trg);
        std::memcpy(ptrg, psrc,
                    sizeof(double) *
                        std::min(num_doubles_src, num_doubles_trg));
      }
      deallocate_buffer(m_data);
    }
    _capacity = num_elements;
    m_data = ptr;
    m_storage = std::move(new_storage);
  }

  /** @brief Copy or accumulate a general expression into the current matrix.
   *
   *  This helper is the "engine" behind expression materialization and generic
   *  `operator+=`.
   *
   *  It walks the storage slice-by-slice and applies either:
   *  - direct assignment (`Equal`)
   *  - in-place accumulation (`EqAdd`)
   *
   *  depending on the compile-time operator tag @p Op.
   *
   *  @tparam Op Reduction mode.
   *  @tparam T Expression type.
   *  @param[in] rhs Expression to reduce into this matrix.
   *
   *  @remark
   *  The traversal is specialized for row-major vs column-major storage so that
   *  the inner loop walks contiguous slices.
   */
  template <detail::ReductionAssignmentOperator Op, typename T>
  void reduce_copy(const T &rhs) {
    /* copy data from rhs to lhs;
     * number of rows = rhs.rows()
     * number of columns = rhs.cols()
     */
    const int cprows = rhs.rows();
    const int cpcols = rhs.cols();
    int num_elements = 0;
    if (!(rows() >= cprows && cols() >= cpcols)) {
      throw std::runtime_error(
          "[ERROR] Failed applying reduce_copy; sizes do not match!\n");
    }

    if constexpr (StorageImplementation<S>::isRowMajor) {
      for (int i = 0; i < cprows; i++) {
        double *entries = slice(i, num_elements);
        for (int j = 0; j < std::min(cpcols, num_elements); j++) {
          detail::op<Op>(entries[j], rhs(i, j));
        }
      }
    } else {
      for (int i = 0; i < cpcols; i++) {
        double *entries = slice(i, num_elements);
        const int k = StorageImplementation<S>::first_row_of_col(i);
        for (int j = k; j < cprows; j++) {
          detail::op<Op>(entries[j - k], rhs(j, i));
        }
      }
    }
  }

  /** @brief Expression node representing a reduced top-left view of another
   *         expression.
   *
   *  @tparam Expr Underlying expression type, either stored by reference or by
   *          value depending on how the view was created.
   *
   *  A reduced view does **not** allocate and does not copy matrix contents.
   *  It simply presents the top-left `reduced_rows x reduced_cols` logical
   *  subexpression of an existing expression.
   *
   *  ## Why this helper exists
   *
   *  It lets the user write expressions such as:
   *
   *  @code
   *  auto rv = A.reduced_view(10, 5);
   *  auto expr = 2.0 * A.reduced_view(20, 20);
   *  auto cropped = (A + B).reduced_view(100, 100);
   *  @endcode
   *
   *  without materializing a new matrix.
   *
   *  ## Ownership model
   *
   *  - If the reduced view is created from an lvalue matrix/expression, the
   *    proxy typically stores a reference.
   *  - If it is created from an rvalue expression proxy, the proxy may own that
   *    small expression object by value.
   *
   *  This allows:
   *
   *  @code
   *  auto rv = (A + B).reduced_view(10, 10);
   *  @endcode
   *
   *  without dangling references to a temporary sum proxy.
   */
  template <typename Expr> class _ReducedViewProxy {
    Expr expr;                  /* any Expression */
    int reduced_rows;           /* reduced rows */
    int reduced_cols;           /* reduced cols */
    friend class CoeffMatrix2D; /* allow CoeffMatrix2D to make views */

    /** @brief Construct a reduced view proxy.
     *  @param[in] e Underlying expression.
     *  @param[in] r Requested reduced row count.
     *  @param[in] c Requested reduced column count.
     *
     *  @throws std::runtime_error if the requested reduced shape exceeds the
     *          source expression dimensions or violates the square-storage
     * rule.
     */
    _ReducedViewProxy(Expr e, int r, int c)
        : expr(std::move(e)), reduced_rows(r), reduced_cols(c) {
      if (!((expr.rows() >= reduced_rows) && (expr.cols() >= reduced_cols))) {
        throw std::runtime_error("[ERROR] Failed to construct  "
                                 "_ReducedViewProxy of given dimensions!\n");
      }
      /* Square matrices must have rows == cols */
      if constexpr (Storage::isSquare) {
        if (reduced_rows != reduced_cols) {
          throw std::runtime_error("[ERROR] Cannot apply a non-square "
                                   "reduced_vew to a square Matrix!\n");
        }
      }
    }

  public:
    /* cannot assert a Contiguous memory layout! */
    /** @brief Reduced views are treated as non-contiguous expressions.
     *
     *  Even if the underlying source is contiguous, the reduced view is modeled
     *  conservatively as a general expression. This keeps the implementation
     *  simple and routes materialization through the generic `reduce_copy`
     *  machinery when needed.
     */
    static constexpr const int hasContiguousMem = 0;
    /** @brief Logical row count of the reduced view. */
    int rows() const noexcept { return reduced_rows; }
    /** @brief Logical column count of the reduced view. */
    int cols() const noexcept { return reduced_cols; }
    /** @brief Access the coefficient at `(i,j)` inside the reduced view.
     *  @return `expr(i,j)` from the underlying expression.
     */
    double operator()(int i, int j) const noexcept { return expr(i, j); }
  };

  /** @brief Expression node representing a scalar multiple of another
   * expression.
   *
   *  @tparam Expr Underlying expression type, stored either by reference or by
   *          value depending on the originating operator call.
   *
   *  This helper exists so that expressions such as:
   *
   *  @code
   *  auto sp1 = 2.0 * A;
   *  auto sp2 = A * 0.5;
   *  auto expr = 1.5 * A.reduced_view(10, 10);
   *  auto expr2 = 0.25 * (A + B);
   *  @endcode
   *
   *  can be represented lazily without allocating a temporary matrix.
   *
   *  @note
   *  If the underlying expression is contiguous in flat storage, the scaled
   *  proxy is also considered contiguous and can expose `data(i)`.
   */
  template <typename Expr> class _ScaledProxy {
    friend class CoeffMatrix2D;
    Expr expr;
    double fac;
    /** @brief Construct a scaled expression proxy.
     *  @param[in] e Underlying expression.
     *  @param[in] d Scalar factor.
     */
    _ScaledProxy(Expr e, double d) noexcept : expr(std::move(e)), fac(d) {};

  public:
    /** @brief Whether the scaled expression preserves flat contiguous access.
     *
     *  This is true iff the underlying expression itself is flat-contiguous.
     */
    static constexpr const int hasContiguousMem =
        std::remove_reference_t<Expr>::hasContiguousMem;
    /** @brief Logical row count of the scaled expression. */
    int rows() const noexcept { return expr.rows(); }
    /** @brief Logical column count of the scaled expression. */
    int cols() const noexcept { return expr.cols(); }
    /** @brief Access `fac * expr(i,j)`. */
    double operator()(int i, int j) const noexcept { return expr(i, j) * fac; }
    /** @brief Access `fac * expr.data(i)` in flat contiguous storage.
     *
     *  This member is only semantically valid when
     *  @ref hasContiguousMem is true.
     */
    double data(int i) const noexcept { return expr.data(i) * fac; }
  }; /* _ScaledProxy */

  /** @brief Expression node representing the sum of two expressions.
   *
   *  @tparam L Left expression storage type.
   *  @tparam R Right expression storage type.
   *
   *  This helper exists so that expressions such as:
   *
   *  @code
   *  auto sp = A + B;
   *  auto expr = 0.5 * A + 2.0 * B;
   *  auto expr2 = A.reduced_view(10, 10) + B.reduced_view(10, 10);
   *  @endcode
   *
   *  can remain lazy and allocation-free until materialized into an actual
   *  matrix.
   *
   *  The proxy is also the key building block for chained arithmetic
   *  expressions. For example:
   *
   *  @code
   *  auto expr = 1.5 * A + 2.0 * B;
   *  CoeffMatrix2D<S> C(expr); // materialize
   *  @endcode
   *
   *  ## Why is this needed?
   *
   *  Without @_SumProxy, a user expression like `A + B` would need to create a
   *  full temporary matrix immediately. That would:
   *  - allocate memory,
   *  - copy all coefficients,
   *  - and defeat the purpose of composing small arithmetic expressions.
   *
   *  `_SumProxy` postpones that cost until the caller explicitly requests a
   *  real matrix.
   */
  template <typename L, typename R> class _SumProxy {
    friend class CoeffMatrix2D;
    L lhs;
    R rhs;
    /** @brief Construct a sum expression.
     *  @param[in] tl Left operand.
     *  @param[in] tr Right operand.
     *
     *  @throws std::runtime_error if the expression dimensions do not match.
     */
    _SumProxy(L tl, R tr) : lhs(std::move(tl)), rhs(std::move(tr)) {
      if (!((lhs.rows() == rhs.rows()) && (lhs.cols() == rhs.cols()))) {
        throw std::runtime_error("[ERROR] Failed constructing _SumProxy cause "
                                 "lhs and rhs sizes do not match!\n");
      }
    }

  public:
    /** @brief Whether both operands support flat contiguous access. */
    static constexpr const int hasContiguousMem =
        std::remove_reference_t<L>::hasContiguousMem &&
        std::remove_reference_t<R>::hasContiguousMem;

    /** @brief Logical row count of the sum expression. */
    int rows() const noexcept { return lhs.rows(); }

    /** @brief Logical column count of the sum expression. */
    int cols() const noexcept { return lhs.cols(); }

    /** @brief Access the coefficient sum at `(i,j)`. */
    double operator()(int i, int j) const noexcept {
      return rhs(i, j) + lhs(i, j);
    }

    /** @brief Access the sum in flat contiguous storage.
     *
     *  This member is only semantically valid when @ref hasContiguousMem is
     *  true.
     */
    double data(int i) const noexcept { return lhs.data(i) + rhs.data(i); }

    /** @brief Create a reduced view from an lvalue sum proxy.
     *
     *  In this case the reduced-view proxy borrows `*this`.
     */
    auto reduced_view(int r, int c) const & noexcept {
      return _ReducedViewProxy<const _SumProxy &>(*this, r, c); // borrows
    }

    /** @brief Create a reduced view from an rvalue sum proxy.
     *
     *  In this case the reduced-view proxy stores the sum proxy by value.
     *  This avoids dangling references when the source is a temporary, e.g.
     *
     *  @code
     *  auto rv = (A + B).reduced_view(10, 10);
     *  @endcode
     */
    auto reduced_view(int r, int c) && noexcept {
      return _ReducedViewProxy<_SumProxy>(std::move(*this), r,
                                          c); // owns the proxy, not a matrix
    }

  }; /* SumProxy */

public:
  /** @brief Swap this matrix with another one.
   *  @param[in,out] b Matrix to swap with.
   *
   *  This swaps:
   *  - storage metadata
   *  - data pointer
   *  - capacity
   *
   *  and is the primitive behind the non-member `swap`.
   */
  void swap(CoeffMatrix2D<S> &b) noexcept {
    using std::swap;
    swap(m_storage, b.m_storage);
    swap(m_data, b.m_data);
    swap(_capacity, b._capacity);
  }

  /** @brief Logical number of rows. */
  constexpr int rows() const noexcept { return m_storage.nrows(); }

  /** @brief Logical number of columns. */
  constexpr int cols() const noexcept { return m_storage.ncols(); }

  /** @brief Number of stored coefficients in the flat memory buffer. */
  constexpr std::size_t num_elements() const noexcept {
    return m_storage.num_elements();
  }

  /** @brief Mutable logical element access.
   *  @param[in] i Row index.
   *  @param[in] j Column index.
   *  @return Reference to the stored coefficient at logical position `(i,j)`.
   *
   *  @warning
   *  The validity of `(i,j)` depends on the storage policy. For compact
   *  triangular/trapezoidal storages, not every logical `(i,j)` pair is valid.
   */
  double &operator()(int i, int j) noexcept {
#ifdef DEBUG
    assert(i < rows() && j < cols());
    assert(m_storage.index(i, j) >= 0 &&
           m_storage.index(i, j) < (int)m_storage.num_elements());
#endif
    return m_data[m_storage.index(i, j)];
  }

  /** @brief Const logical element access.
   *  @param[in] i Row index.
   *  @param[in] j Column index.
   *  @return Const reference to the stored coefficient at logical position
   *          `(i,j)`.
   */
  const double &operator()(int i, int j) const noexcept {
#ifdef DEBUG
    assert(i < rows() && j < cols());
    assert(m_storage.index(i, j) >= 0 &&
           m_storage.index(i, j) < (int)m_storage.num_elements());
#endif
    return m_data[m_storage.index(i, j)];
  }

  /** @brief Return the start pointer of a logical row/column slice.
   *
   *  The meaning depends on the storage policy:
   *  - row-major storage: beginning of row `i`
   *  - column-major storage: beginning of column `i`
   *
   *  @param[in] i Slice index.
   *  @return Pointer into the flat storage buffer.
   *
   *  @see StorageImplementation::slice(int)
   */
  const double *slice(int i) const noexcept {
    return m_data + m_storage.slice(i);
  }

  /** @brief Mutable variant of @ref slice(int) const. */
  double *slice(int i) noexcept { return m_data + m_storage.slice(i); }

  /** @brief Pointer to the beginning of column `j`.
   *
   *  This member exists only for column-major storage policies.
   *
   *  @param[in] j Zero-based column index.
   *  @return Mutable pointer to the contiguous stored column segment.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isColMajor, bool> = true>
  double *column(int j) noexcept {
    return slice(j);
  }

  /** @brief Const pointer to the beginning of column `j`.
   *
   *  This member exists only for column-major storage policies.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isColMajor, bool> = true>
  const double *column(int j) const noexcept {
    return slice(j);
  }

  /** @brief Pointer to the beginning of row `j`.
   *
   *  This member exists only for row-major storage policies.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isRowMajor, bool> = true>
  double *row(int j) noexcept {
    return slice(j);
  }

  /** @brief Const pointer to the beginning of row `j`.
   *
   *  This member exists only for row-major storage policies.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isRowMajor, bool> = true>
  const double *row(int j) const noexcept {
    return slice(j);
  }

  /** @brief Set all stored coefficients to the same value.
   *  @param[in] val Value to write into every stored coefficient.
   */
  void fill_with(double val) noexcept {
    std::fill(m_data, m_data + m_storage.num_elements(), val);
  }

  /** @brief Multiply all stored coefficients by a scalar.
   *  @param[in] value Scalar multiplier.
   */
  void multiply(double value) noexcept {
    std::transform(m_data, m_data + m_storage.num_elements(), m_data,
                   [=](double d) { return d * value; });
  }

  /** @brief Return a const pointer to the underlying contiguous storage. */
  const double *data() const noexcept { return m_data; }

  /** @brief Return a mutable pointer to the underlying contiguous storage. */
  double *data() noexcept { return m_data; }

  /** @brief Expression-template operator for `Expression + Expression`.
   *
   *  This friend operator creates a @_SumProxy instead of materializing a new
   *  matrix immediately.
   *
   *  @tparam A Left expression type.
   *  @tparam B Right expression type.
   *  @param[in] a Left operand.
   *  @param[in] b Right operand.
   *  @return A lazy sum expression.
   *
   *  ## Borrowing vs owning
   *
   *  - if an operand is an lvalue, the proxy stores it by `const&`
   *  - if an operand is an rvalue expression, the proxy stores it by value
   *
   *  This allows efficient and safe expressions such as:
   *
   *  @code
   *  auto sp = A + B;
   *  auto expr = 0.5 * A + 2.0 * B;
   *  auto rv = (A + B).reduced_view(10, 10);
   *  @endcode
   *
   *  while still forbidding temporary *matrices* through the deleted overloads
   *  below.
   */
  template <
      class A, class B,
      std::enable_if_t<detail::_is_expr_v<A> && detail::_is_expr_v<B>, int> = 0>
  friend auto operator+(A &&a, B &&b) noexcept {
    /* if A is an lvalue, just borrow, ExprT -> const A&; if A is an rvalue then
     * we should own the instance A, ExprT -> A
     */
    using ExprTa = std::conditional_t<std::is_lvalue_reference_v<A &&>,
                                      const std::remove_reference_t<A> &,
                                      std::remove_reference_t<A>>;
    using ExprTb = std::conditional_t<std::is_lvalue_reference_v<B &&>,
                                      const std::remove_reference_t<B> &,
                                      std::remove_reference_t<B>>;
    // keep dimension checks in the proxy ctor (preferred).
    return _SumProxy<ExprTa, ExprTb>(std::forward<A>(a), std::forward<B>(b));
  }

  /** @brief Deleted overload forbidding `temporary_matrix + expression`.
   *
   *  This prevents storing a whole temporary matrix inside an expression tree.
   */
  template <class X>
  friend auto operator+(CoeffMatrix2D &&, const X &) = delete;

  /** @brief Deleted overload forbidding `expression + temporary_matrix`. */
  template <class X>
  friend auto operator+(const X &, CoeffMatrix2D &&) = delete;

  /** @brief Expression-template operator for `Expression * scalar`.
   *
   *  This returns a @_ScaledProxy instead of materializing a temporary matrix.
   *
   *  Example:
   *  @code
   *  auto sp = A * 2.0;
   *  auto expr = (A + B) * 0.25;
   *  @endcode
   */
  template <class A, std::enable_if_t<detail::_is_expr_v<A>, int> = 0>
  friend auto operator*(A &&a, double s) noexcept {
    /* if A is an lvalue, just borrow, ExprT -> const A&; if A is an rvalue then
     * we should own the instance A, ExprT -> A
     */
    using ExprT = std::conditional_t<std::is_lvalue_reference_v<A &&>,
                                     const std::remove_reference_t<A> &,
                                     std::remove_reference_t<A>>;
    return _ScaledProxy<ExprT>(std::forward<A>(a), s);
  }

  /** @brief Expression-template operator for `scalar * Expression`.
   *
   *  This is the symmetric overload of `Expression * scalar`.
   */
  template <class A, std::enable_if_t<detail::_is_expr_v<A>, int> = 0>
  friend auto operator*(double s, A &&a) noexcept {
    /* if A is an lvalue, just borrow, ExprT -> const A&; if A is an rvalue then
     * we should own the instance A, ExprT -> A
     */
    using ExprT = std::conditional_t<std::is_lvalue_reference_v<A &&>,
                                     const std::remove_reference_t<A> &,
                                     std::remove_reference_t<A>>;
    return _ScaledProxy<ExprT>(std::forward<A>(a), s);
  }

  /** @brief Deleted overload forbidding `temporary_matrix * scalar`. */
  friend auto operator*(CoeffMatrix2D &&, double) = delete;

  /** @brief Deleted overload forbidding `scalar * temporary_matrix`. */
  friend auto operator*(double, CoeffMatrix2D &&) = delete;

  /** @brief Create a reduced top-left view from an lvalue matrix.
   *  @param[in] r Reduced row count.
   *  @param[in] c Reduced column count.
   *  @return A lazy reduced-view proxy.
   */
  auto reduced_view(int r, int c) const & noexcept {
    return _ReducedViewProxy<const CoeffMatrix2D &>(*this, r, c);
  }

#if __cplusplus >= 202002L /* concepts only available in C++20 */
  /** @brief Square-only shorthand for `reduced_view(rc, rc)`.
   *
   *  This overload exists only for square storage policies.
   *
   *  Example:
   *  @code
   *  CoeffMatrix2D<MatrixStorageType::LwTriangularRowWise> A(20);
   *  auto rv = A.reduced_view(10); // means reduced_view(10,10)
   *  @endcode
   */
  auto reduced_view(int rc) const & noexcept
    requires(Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  auto reduced_view(int rc) {
#endif
#ifdef DEBUG
    assert(rc >= 0 && rc <= rows());
#endif
    return reduced_view(rc, rc);
  }

  /** @brief Deleted overload: reduced views of temporary matrices are
   * forbidden. */
  auto reduced_view(int, int) const && = delete; /* no temporary matrices */
  /** @brief Deleted overload: reduced views of temporary square matrices are
   * forbidden. */
  auto reduced_view(int) const && = delete; /* no temporary matrices */

  /** @brief Construct a non-square matrix from row and column counts.
   *
   *  This overload exists only for non-square storage policies.
   */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  CoeffMatrix2D(int rows, int cols)
    requires(!Storage::isSquare)
#else
  template <bool B = Storage::isSquare, std::enable_if_t<!B, int> = 0>
  CoeffMatrix2D(int rows, int cols)
#endif
      : m_storage(make_storage_impl(rows, cols)),
        m_data((m_storage.num_elements() > 0)
                   ? (allocate_buffer(m_storage.num_elements()))
                   : (nullptr)),
        _capacity(m_storage.num_elements()) {
#ifdef DEBUG
    assert(m_storage.num_elements() >= 0);
#endif
  };

#if __cplusplus >= 202002L /* concepts only available in C++20 */
  /** @brief Construct a square matrix from a single dimension.
   *
   *  This overload exists only for square storage policies.
   */
  CoeffMatrix2D(int rows)
    requires(Storage::isSquare)
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  CoeffMatrix2D(int rows)
#endif
      : m_storage(rows),
        m_data((m_storage.num_elements() > 0)
                   ? (allocate_buffer(m_storage.num_elements()))
                   : (nullptr)),
        _capacity(m_storage.num_elements()) {
#ifdef DEBUG
    assert(m_storage.num_elements() >= 0);
#endif
  };

  /** @brief Destructor. Releases the owned storage buffer. */
  ~CoeffMatrix2D() noexcept {
    deallocate_buffer(m_data);
    _capacity = 0;
  }

  /** @brief Copy constructor.
   *  @param[in] mat Source matrix.
   *
   *  The new matrix receives its own storage buffer and copies all stored
   *  coefficients from the source.
   */
  CoeffMatrix2D(const CoeffMatrix2D &mat) noexcept
      : m_storage(mat.m_storage), m_data(allocate_buffer(mat.num_elements())),
        _capacity(mat.num_elements()) {
    std::memcpy(m_data, mat.m_data, sizeof(double) * mat.num_elements());
  }

  /** @brief Move constructor.
   *  @param[in,out] mat Source matrix to steal from.
   *
   *  After the move, the source matrix is left empty.
   */
  CoeffMatrix2D(CoeffMatrix2D &&mat) noexcept
      : m_storage(mat.m_storage), m_data(mat.m_data), _capacity(mat._capacity) {
    mat.m_data = nullptr;
    if constexpr (Storage::isSquare)
      mat.m_storage.__set_dimensions(0);
    else
      mat.m_storage.__set_dimensions(0, 0);
    mat._capacity = 0;
  }

  /** @brief Materialize a sum expression into a real matrix.
   *
   *  This constructor lets users write:
   *  @code
   *  CoeffMatrix2D<S> C(A + B);
   *  auto expr = 0.5 * A + 2.0 * B;
   *  CoeffMatrix2D<S> D(expr);
   *  @endcode
   *
   *  If the sum expression is contiguous in flat storage, the constructor uses
   *  the fast `data(i)` path. Otherwise it falls back to
   *  @ref reduce_copy.
   */
  template <typename L, typename R>
  CoeffMatrix2D(const _SumProxy<L, R> &sum) noexcept
      : m_storage(make_storage_impl(sum.rows(), sum.cols())),
        m_data(allocate_buffer(m_storage.num_elements())),
        _capacity(m_storage.num_elements()) {
    if constexpr (_SumProxy<L, R>::hasContiguousMem) {
      for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
        m_data[i] = sum.data(i);
      }
    } else {
      reduce_copy<detail::ReductionAssignmentOperator::Equal>(sum);
    }
  }

  /** @brief Rvalue overload for sum-expression materialization. */
  template <typename L, typename R>
  CoeffMatrix2D(_SumProxy<L, R> &&sum) noexcept
      : m_storage(make_storage_impl(sum.rows(), sum.cols())),
        m_data(allocate_buffer(m_storage.num_elements())),
        _capacity(m_storage.num_elements()) {
    if constexpr (_SumProxy<L, R>::hasContiguousMem) {
      for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
        m_data[i] = sum.data(i);
      }
    } else {
      reduce_copy<detail::ReductionAssignmentOperator::Equal>(sum);
    }
  }

  /** @brief Materialize a scaled expression into a real matrix (lvalue source).
   */
  template <typename T1>
  CoeffMatrix2D(const _ScaledProxy<T1> &fac) noexcept
      : m_storage(make_storage_impl(fac.rows(), fac.cols())),
        m_data(allocate_buffer(m_storage.num_elements())),
        _capacity(m_storage.num_elements()) {
    if constexpr (_ScaledProxy<T1>::hasContiguousMem) {
      for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
        m_data[i] = fac.data(i);
      }
    } else {
      reduce_copy<detail::ReductionAssignmentOperator::Equal>(fac);
    }
  }

  /** @brief Materialize a scaled expression into a real matrix (rvalue source).
   */
  template <typename T1>
  CoeffMatrix2D(_ScaledProxy<T1> &&fac) noexcept
      : CoeffMatrix2D(static_cast<const _ScaledProxy<T1> &>(fac)) {};

  /** @brief Copy assignment.
   *  @param[in] mat Source matrix.
   *  @return `*this`
   */
  CoeffMatrix2D &operator=(const CoeffMatrix2D &mat) noexcept {
    if (this != &mat) {
      /* do we need extra capacity ? */
      if (_capacity < mat._capacity) {
        if (m_data)
          deallocate_buffer(m_data);
        m_data = allocate_buffer(mat.num_elements());
        _capacity = mat.num_elements();
      }
      std::memcpy(m_data, mat.m_data, sizeof(double) * mat.num_elements());
      if constexpr (Storage::isSquare)
        m_storage.__set_dimensions(mat.rows());
      else
        m_storage.__set_dimensions(mat.rows(), mat.cols());
    }
    return *this;
  }

  /** @brief Move assignment.
   *  @param[in,out] mat Source matrix to steal from.
   *  @return `*this`
   */
  CoeffMatrix2D &operator=(CoeffMatrix2D &&mat) noexcept {
    if (this != &mat) {
      if (m_data)
        deallocate_buffer(m_data);
      m_storage = mat.m_storage;
      m_data = mat.m_data;
      _capacity = mat._capacity;

      mat.m_data = nullptr;
      if constexpr (Storage::isSquare)
        mat.m_storage.__set_dimensions(0);
      else
        mat.m_storage.__set_dimensions(0, 0);
      mat._capacity = 0;
    }
    return *this;
  }

  /** @brief Resize without preserving values (square overload). */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void resize(int rows)
    requires(Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  void resize(int rows) {
#endif
    return resize_impl(rows, rows);
  }

  /** @brief Resize without preserving values (rectangular overload). */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void resize(int rows, int cols)
    requires(!Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<!B, int> = 0>
  void resize(int rows, int cols) {
#endif
    return resize_impl(rows, cols);
  }

  /** @brief Resize while preserving overlapping values (rectangular overload).
   */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void cresize(int rows, int cols)
    requires(!Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<!B, int> = 0>
  void cresize(int rows, int cols) {
#endif
    return cresize_impl(rows, cols);
  }

  /** @brief Resize while preserving overlapping values (square overload). */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void cresize(int rows)
    requires(Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  void cresize(int rows) {
#endif
      return cresize_impl(rows, rows);
}

/** @brief Generic in-place accumulation from an expression.
 *
 *  This is the generic fallback for:
 *  @code
 *  A += rhs;
 *  @endcode
 *
 *  where `rhs` is any valid matrix expression.
 *
 *  If the expression is contiguous, flat storage is used directly. Otherwise
 *  the operation falls back to @ref reduce_copy.
 */
template <typename T, std::enable_if_t<detail::_is_expr_v<T>, int> = 0>
CoeffMatrix2D &operator+=(const T &rhs) {
  if constexpr (T::hasContiguousMem) {
    if (!((this->rows() == rhs.rows()) && (this->cols() == rhs.cols()))) {
      throw std::runtime_error("[ERROR] Invalid matrix dimensions for "
                               "CoeffMatrix2D::operator+=\n");
    }
    for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
      m_data[i] += rhs.data(i);
    }
  } else {
    reduce_copy<detail::ReductionAssignmentOperator::EqAdd>(rhs);
  }
  return *this;
}

/* Specific for LwTriangularColWise:
 * Following are optimizations targeting MatrixStorageType::LwTriangularColWise
 * leveraging SIMD operations
 *
 * axpy: operation of type A = A + s * B where B is LwTriangularColWise and s is
 *       a scalar
 */

/** @brief Optimized in-place AXPY kernel for
 *         `MatrixStorageType::LwTriangularColWise`.
 *
 *  Performs:
 *  @code
 *  this += s * rhs;
 *  @endcode
 *
 *  but through the dedicated low-level kernel rather than the generic
 *  expression-template path.
 *
 *  @note
 *  This member exists only when `S == MatrixStorageType::LwTriangularColWise`.
 */
#if __cplusplus >= 202002L
void axpy_inplace(double s, const CoeffMatrix2D &rhs) noexcept
  requires(S == MatrixStorageType::LwTriangularColWise)
#else
  template <
      MatrixStorageType SS = S,
      std::enable_if_t<SS == MatrixStorageType::LwTriangularColWise, int> = 0>
  void axpy_inplace(double s, const CoeffMatrix2D &rhs) noexcept
#endif
{
#ifdef DEBUG
  assert(this->rows() == rhs.rows());
  assert(this->cols() == rhs.cols());
#endif
  detail::axpy_lwtri_colwise(m_data, rhs.m_data, s, m_storage.num_elements());
}

/** @brief Specialized `operator+=` overload for `A += s * B` when the storage
 *         is lower-triangular column-wise.
 *
 *  This overload routes the operation directly to the SIMD/scalar AXPY kernel,
 *  bypassing the generic element-wise expression traversal.
 */
#if __cplusplus >= 202002L
CoeffMatrix2D &
operator+=(const _ScaledProxy<const CoeffMatrix2D &> &rhs) noexcept
  requires(S == MatrixStorageType::LwTriangularColWise)
#else
  template <
      MatrixStorageType SS = S,
      std::enable_if_t<SS == MatrixStorageType::LwTriangularColWise, int> = 0>
  CoeffMatrix2D &
  operator+=(const _ScaledProxy<const CoeffMatrix2D &> &rhs) noexcept
#endif
{
#ifdef DEBUG
  assert(this->rows() == rhs.rows());
  assert(this->cols() == rhs.cols());
#endif
  detail::axpy_lwtri_colwise(m_data, rhs.expr.m_data, rhs.fac,
                             m_storage.num_elements());
  return *this;
}

/** @brief Optimized in-place two-source AXPY kernel for
 *         `MatrixStorageType::LwTriangularColWise`.
 *
 *  Performs:
 *  @code
 *  this += s1 * B1 + s2 * B2;
 *  @endcode
 *
 *  through the dedicated low-level kernel @ref detail::axpy2_lwtri_colwise.
 *
 *  This operation is intentionally exposed as an explicit member rather than
 *  implicitly overloading `operator+=` for the corresponding sum-of-scaled
 *  expression shape. That keeps the interface explicit and avoids adding more
 *  overload-selection complexity.
 *
 *  @warning
 *  `this`, `B1`, and `B2` are expected to refer to distinct non-overlapping
 *  matrices, because the low-level kernel uses `DSO_RESTRICT`.
 */
#if __cplusplus >= 202002L
void axpy2_inplace(double s1, const CoeffMatrix2D &B1, double s2,
                   const CoeffMatrix2D &B2) noexcept
  requires(S == MatrixStorageType::LwTriangularColWise)
#else
  template <
      MatrixStorageType SS = S,
      std::enable_if_t<SS == MatrixStorageType::LwTriangularColWise, int> = 0>
  void axpy2_inplace(double s1, const CoeffMatrix2D &B1, double s2,
                     const CoeffMatrix2D &B2) noexcept
#endif
{
#ifdef DEBUG
  assert(this->rows() == B1.rows());
  assert(this->cols() == B1.cols());
  assert(this->rows() == B2.rows());
  assert(this->cols() == B2.cols());
#endif

  detail::axpy2_lwtri_colwise(m_data, B1.m_data, B2.m_data, s1, s2,
                              m_storage.num_elements());
}

}; // namespace dso

} /* namespace dso */

/** @brief Non-member swap forwarding to @ref dso::CoeffMatrix2D::swap.
 *  @tparam S Storage policy.
 *  @param[in,out] a First matrix.
 *  @param[in,out] b Second matrix.
 */
template <dso::MatrixStorageType S>
inline void swap(dso::CoeffMatrix2D<S> &a, dso::CoeffMatrix2D<S> &b) noexcept {
  a.swap(b);
}

#endif
