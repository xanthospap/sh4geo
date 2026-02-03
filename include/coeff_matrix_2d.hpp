/** @file
 * Naive implementation of 2-Dimensional matrices; note that this
 * implementation targets 2-dimensional matrices that are only meant to store
 * data NOT perform arithmetic operations.
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

namespace {
inline double &op_equal(double &lhs, double rhs) noexcept { return lhs = rhs; }
inline double &op_eqadd(double &lhs, double rhs) noexcept { return lhs += rhs; }
enum class ReductionAssignmentOperator { Equal, EqAdd };
template <ReductionAssignmentOperator Op>
inline double &op(double &lhs, double rhs) noexcept {
  if constexpr (Op == ReductionAssignmentOperator::Equal)
    return op_equal(lhs, rhs);
  else
    return op_eqadd(lhs, rhs);
}

/* An expression is usually nothing ... */
template <class T, class = void> struct _is_expr : std::false_type {};

/* ...except if it has rows(), cols() and an operator(int i, int j) !! */
template <class T>
struct _is_expr<T, std::void_t<decltype(std::declval<const T &>().rows()),
                               decltype(std::declval<const T &>().cols()),
                               decltype(std::declval<const T &>()(0, 0))>>
    : std::true_type {};

/* utility class expose for _is_expr */
template <class T>
static constexpr bool _is_expr_v =
    _is_expr<std::remove_cv_t<std::remove_reference_t<T>>>::value;

} // namespace

namespace dso {

/** @brief A naive implementation of a 2-d dense matrix.
 *  The main objective of this class is to store data (e.g. coefficients) and
 *  not perform arithmetic operations.
 */
template <MatrixStorageType S> class CoeffMatrix2D {
private:
  using Storage = StorageImplementation<S>;

  Storage m_storage;        /** storage type; dictates indexing */
  double *m_data{nullptr};  /** the actual data */
  std::size_t _capacity{0}; /** number of doubles in allocated memory arena */
  static constexpr const int hasContiguousMem = true;

  /** Access an element from the underlying data; use with care IF needed */
  double data(int i) const noexcept { return m_data[i]; }

  /** @brief Row/Column indexing (rows and columns start from 0 --not 1--)
   *
   * If the data is stored in a Row-Wise manner, this function will return
   * the offset of the ith row; if data is stored in a column-wise manner,
   * it will return the index of the first elelement of the ith column.
   *
   * The parameter \p num_elements will hold the number of elements stored
   * in this row/col.
   */
  const double *slice(int i, int &num_elements) const noexcept {
    return m_data + m_storage.slice(i, num_elements);
  }

  /** @brief Row/Column indexing (rows and columns start from 0 --not 1--)
   *
   * If the data is stored in a Row-Wise manner, this function will return
   * the offset of the ith row; if data is stored in a column-wise manner,
   * it will return the index of the first elelement of the ith column.
   *
   * The parameter \p num_elements will hold the number of elements stored
   * in this row/col.
   */
  double *slice(int i, int &num_elements) noexcept {
    return m_data + m_storage.slice(i, num_elements);
  }

  template <ReductionAssignmentOperator Op, typename T>
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
          op<Op>(entries[j], rhs(i, j));
        }
      }
    } else {
      for (int i = 0; i < cpcols; i++) {
        double *entries = slice(i, num_elements);
        const int k = StorageImplementation<S>::first_row_of_col(i);
        for (int j = k; j < cprows; j++) {
          op<Op>(entries[j - k], rhs(j, i));
        }
      }
    }
  }

  /* T1 can be e.g. a CoeffMatrix2D<...>, _ScaledProxy<...>, _SumProxy<...> */
  template <typename Expr> class _ReducedViewProxy {
    Expr expr;                  /* any Expression */
    int reduced_rows;           /* reduced rows */
    int reduced_cols;           /* reduced cols */
    friend class CoeffMatrix2D; /* allow CoeffMatrix2D to make views */

    _ReducedViewProxy(Expr e, int r, int c)
        : expr(std::move(e)), reduced_rows(r), reduced_cols(c) {
      if (!((expr.rows() >= reduced_rows) && (expr.cols() >= reduced_cols))) {
        throw std::runtime_error("[ERROR] Failed to construct  "
                                 "_ReducedViewProxy of given dimensions!\n");
      }
      /* Square matrices must have rows == cols */
      if constexpr (Storage::isSquare) {
        if (expr.rows() != expr.cols()) {
          throw std::runtime_error("[ERROR] Cannot apply a non-square "
                                   "reduced_vew to a square Matrix!\n");
        }
      }
    }

  public:
    /* cannot assert a Contiguous memory layout! */
    static constexpr const int hasContiguousMem = 0;
    /* reduced rows */
    int rows() const noexcept { return reduced_rows; }
    /* reduced cols */
    int cols() const noexcept { return reduced_cols; }
    /* (const) access operator, given (row, col) */
    double operator()(int i, int j) const noexcept { return expr(i, j); }
  };

  /** Expression Template: Structure to hold a scaled CoeffMatrix2D (i.e. the
   * multiplication of some a matrix by a real number.
   */
  template <typename Expr> class _ScaledProxy {
    friend class CoeffMatrix2D;
    Expr expr;
    double fac;
    _ScaledProxy(Expr e, double d) noexcept : expr(std::move(e)), fac(d) {};

  public:
    static constexpr const int hasContiguousMem =
        std::remove_reference_t<Expr>::hasContiguousMem;
    int rows() const noexcept { return expr.rows(); }
    int cols() const noexcept { return expr.cols(); }
    /* (const) access operator, gives Expr(row, col) * scalar */
    double operator()(int i, int j) const noexcept { return expr(i, j) * fac; }
    /* (const) access operator, gives data(index) * scalar */
    double data(int i) const noexcept { return expr.data(i) * fac; }
  }; /* _ScaledProxy */

  template <typename L, typename R> class _SumProxy {
    friend class CoeffMatrix2D;
    L lhs;
    R rhs;
    _SumProxy(L tl, R tr) : lhs(std::move(tl)), rhs(std::move(tr)) {
      if (!((lhs.rows() == rhs.rows()) && (lhs.cols() == rhs.cols()))) {
        throw std::runtime_error("[ERROR] Failed constructing _SumProxy cause "
                                 "lhs and rhs sizes do not match!\n");
      }
    }

  public:
    static constexpr const int hasContiguousMem =
        std::remove_reference_t<L>::hasContiguousMem &&
        std::remove_reference_t<R>::hasContiguousMem;

    int rows() const noexcept { return lhs.rows(); }
    int cols() const noexcept { return lhs.cols(); }
    double operator()(int i, int j) const noexcept {
      return rhs(i, j) + lhs(i, j);
    }
    double data(int i) const noexcept { return lhs.data(i) + rhs.data(i); }

    auto reduced_view(int r, int c) const & noexcept {
      return _ReducedViewProxy<const _SumProxy &>(*this, r, c); // borrows
    }
    auto reduced_view(int r, int c) && noexcept {
      return _ReducedViewProxy<_SumProxy>(std::move(*this), r,
                                          c); // owns the proxy, not a matrix
    }

  }; /* SumProxy */

public:
  /** Swap current instance with another */
  void swap(CoeffMatrix2D<S> &b) noexcept {
    using std::swap;
    StorageImplementation<S> tmp_s = b.m_storage;
    std::size_t tmp_c = b._capacity;

    swap(m_data, b.m_data);
    b.m_storage = this->m_storage;
    b._capacity = this->_capacity;
    this->m_storage = tmp_s;
    this->_capacity = tmp_c;

    return;
  }

  /** get number of rows */
  constexpr int rows() const noexcept { return m_storage.nrows(); }

  /** get number of columns */
  constexpr int cols() const noexcept { return m_storage.ncols(); }

  /** get number of elements */
  constexpr std::size_t num_elements() const noexcept {
    return m_storage.num_elements();
  }

  /** @brief Element indexing (rows and columns start from 0 --not 1--) */
  double &operator()(int i, int j) noexcept {
#ifdef DEBUG
    assert(i < rows() && j < cols());
    assert(m_storage.index(i, j) >= 0 &&
           m_storage.index(i, j) < (int)m_storage.num_elements());
#endif
    return m_data[m_storage.index(i, j)];
  }

  /** @brief Element indexing (rows and columns start from 0 --not 1--) */
  const double &operator()(int i, int j) const noexcept {
#ifdef DEBUG
    assert(i < rows() && j < cols());
    assert(m_storage.index(i, j) >= 0 &&
           m_storage.index(i, j) < (int)m_storage.num_elements());
#endif
    return m_data[m_storage.index(i, j)];
  }

  /** @brief Row/Column indexing (rows and columns start from 0 --not 1--)
   *
   * If the data is stored in a Row-Wise manner, this function will return
   * the offset of the ith row; if data is stored in a column-wise manner,
   * it will return the index of the first elelement of the ith column.
   */
  const double *slice(int i) const noexcept {
    return m_data + m_storage.slice(i);
  }

  /** @brief Row/Column indexing (rows and columns start from 0 --not 1--)
   *
   * If the data is stored in a Row-Wise manner, this function will return
   * the offset of the ith row; if data is stored in a column-wise manner,
   * it will return the index of the first elelement of the ith column.
   */
  double *slice(int i) noexcept { return m_data + m_storage.slice(i); }

  /** @brief Pointer to the begining of a given column.
   *
   * Only defined if the MatrixStorageType uses some kind of Column Major
   * storage sequence.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isColMajor, bool> = true>
  double *column(int j) noexcept {
    return slice(j);
  }

  /** @brief Const pointer to the begining of a given column.
   *
   * Only defined if the MatrixStorageType uses some kind of Column Major
   * storage sequence.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isColMajor, bool> = true>
  const double *column(int j) const noexcept {
    return slice(j);
  }

  /** @brief Pointer to the begining of a given row.
   *
   * Only defined if the MatrixStorageType uses some kind of Row Major
   * storage sequence.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isRowMajor, bool> = true>
  double *row(int j) noexcept {
    return slice(j);
  }

  /** @brief Const pointer to the begining of a given row.
   *
   * Only defined if the MatrixStorageType uses some kind of Row Major
   * storage sequence.
   */
  template <MatrixStorageType t = S,
            std::enable_if_t<StorageImplementation<t>::isRowMajor, bool> = true>
  const double *row(int j) const noexcept {
    return slice(j);
  }

  /** set all elements of the matrix equal to some value */
  void fill_with(double val) noexcept {
    std::fill(m_data, m_data + _capacity /*m_storage.num_elements()*/, val);
  }

  /** multiply all elements of matrix with given value */
  void multiply(double value) noexcept {
    std::transform(m_data, m_data + m_storage.num_elements(), m_data,
                   [=](double d) { return d * value; });
  }

  /** get a pointer to the data */
  const double *data() const noexcept { return m_data; }

  /** get a non-const pointer to the data */
  double *data() noexcept { return m_data; }

  /* Expression + Expression = _SumProxy */
  template <class A, class B,
            std::enable_if_t<_is_expr_v<A> && _is_expr_v<B>, int> = 0>
  friend auto operator+(const A &a, const B &b) noexcept {
    // keep dimension checks in the proxy ctor (preferred).
    return _SumProxy<const A &, const B &>(a, b);
  }

  /* Not allowed: temporary + Expression */
  template <class X>
  friend auto operator+(CoeffMatrix2D &&, const X &) = delete;

  /* Not allowed: Expression + temporary */
  template <class X>
  friend auto operator+(const X &, CoeffMatrix2D &&) = delete;

  /* Expression * scalar = _ScaledProxy */
  template <class A, std::enable_if_t<_is_expr_v<A>, int> = 0>
  friend auto operator*(const A &a, double s) noexcept {
    return _ScaledProxy<const A &>(a, s);
  }

  /* scalar * Expression = _ScaledProxy */
  template <class A, std::enable_if_t<_is_expr_v<A>, int> = 0>
  friend auto operator*(double s, const A &a) noexcept {
    return _ScaledProxy<const A &>(a, s);
  }

  /* Not allowed: temporary * scalar */
  friend auto operator*(CoeffMatrix2D &&, double) = delete;

  /* Not allowed: scalar * temporary */
  friend auto operator*(double, CoeffMatrix2D &&) = delete;

  auto reduced_view(int r, int c) const & noexcept {
    return _ReducedViewProxy<const CoeffMatrix2D &>(*this, r, c);
  }

  auto reduced_view(int, int) const && = delete; // no temporary matrices

  /** Constructor using number of rows and columns; for some
   * MatrixStorageType's, the number of columns may not be needed.
   */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  CoeffMatrix2D(int rows, int cols)
    requires(!Storage::isSquare)
#else
  template <bool B = Storage::isSquare, std::enable_if_t<!B, int> = 0>
  CoeffMatrix2D(int rows, int cols)
#endif
      : m_storage(rows, cols),
        m_data((m_storage.num_elements() > 0)
                   ? (new double[m_storage.num_elements()])
                   : (nullptr)),
        _capacity(m_storage.num_elements()) {
#ifdef DEBUG
    assert(m_storage.num_elements() >= 0);
#endif
  };

#if __cplusplus >= 202002L /* concepts only available in C++20 */
  CoeffMatrix2D(int rows)
    requires(Storage::isSquare)
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  CoeffMatrix2D(int rows, int cols)
#endif
      : m_storage(rows), m_data((m_storage.num_elements() > 0)
                                    ? (new double[m_storage.num_elements()])
                                    : (nullptr)),
        _capacity(m_storage.num_elements()) {
#ifdef DEBUG
    assert(m_storage.num_elements() >= 0);
#endif
  };

  /** Destructor; free memmory */
  ~CoeffMatrix2D() noexcept {
    if (m_data)
      delete[] m_data;
    _capacity = 0;
  }

  /** Copy constructor */
  CoeffMatrix2D(const CoeffMatrix2D &mat) noexcept
      : m_storage(mat.m_storage), m_data(new double[mat.num_elements()]),
        _capacity(mat.num_elements()) {
    std::memcpy(m_data, mat.m_data, sizeof(double) * mat.num_elements());
  }

  /** Move constructor */
  CoeffMatrix2D(CoeffMatrix2D &&mat) noexcept
      : m_storage(mat.m_storage), m_data(mat.m_data), _capacity(mat._capacity) {
    mat.m_data = nullptr;
    if constexpr (Storage::isSquare)
      mat.m_storage.__set_dimensions(0);
    else
      mat.m_storage.__set_dimensions(0, 0);
    mat._capacity = 0;
  }

  template <typename L, typename R>
  CoeffMatrix2D(_SumProxy<L, R> &&sum) noexcept
      : m_storage(sum.rows(), sum.cols()),
        m_data(new double[m_storage.num_elements()]),
        _capacity(m_storage.num_elements()) {
    if constexpr (_SumProxy<L, R>::hasContiguousMem) {
      for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
        m_data[i] = sum.data(i);
      }
    } else {
      reduce_copy<ReductionAssignmentOperator::Equal>(sum);
    }
  }

  /* Constructor from a _ScaledProxy lvalue */
  template <typename T1>
  CoeffMatrix2D(const _ScaledProxy<T1> &fac) noexcept
      : m_storage(fac.rows(), fac.cols()),
        m_data(new double[m_storage.num_elements()]),
        _capacity(m_storage.num_elements()) {
    if constexpr (_ScaledProxy<T1>::hasContiguousMem) {
      for (std::size_t i = 0; i < m_storage.num_elements(); i++) {
        m_data[i] = fac.data(i);
      }
    } else {
      reduce_copy<ReductionAssignmentOperator::Equal>(fac);
    }
  }

  /* Constructor from a _ScaledProxy rvalue */
  template <typename T1>
  CoeffMatrix2D(_ScaledProxy<T1> &&fac) noexcept
      : CoeffMatrix2D(static_cast<const _ScaledProxy<T1> &>(fac)){};

  /** (Copy) Assignment operator */
  CoeffMatrix2D &operator=(const CoeffMatrix2D &mat) noexcept {
    if (this != &mat) {
      /* do we need extra capacity ? */
      if (_capacity < mat._capacity) {
        if (m_data)
          delete[] m_data;
        m_data = new double[mat.num_elements()];
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

  /** Move Assignment operator */
  CoeffMatrix2D &operator=(CoeffMatrix2D &&mat) noexcept {
    if (this != &mat) {
      if (m_data)
        delete[] m_data;
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

  /** @brief Resize (keeping the MatrixStorageType the same).
   *
   * Note that calling this function may incur data loss, since we are
   * resizing (re-allocating) but not copying the data already stored within
   * the data structure. That is any element A(i,j) of the matrix after
   * calling this function, is not guranted to have the same value as the one
   * it had before calling the function.
   *
   * If you need to resize but also keep the values already stored in the
   * instance, then you should better call the cresize (member) function.
   */
#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void resize(int rows)
    requires(!Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<B, int> = 0>
  void resize(int rows) {
#endif
    /* do we need to re-allocate ? */
    if (Storage(rows).num_elements() > _capacity) {
      if (m_data)
        delete[] m_data;
      m_data = new double[Storage(rows).num_elements()];
      _capacity = Storage(rows).num_elements();
    }
    m_storage = Storage(rows);
  }

#if __cplusplus >= 202002L /* concepts only available in C++20 */
  void resize(int rows, int cols)
    requires(!Storage::isSquare)
  {
#else
  template <bool B = Storage::isSquare, std::enable_if_t<!B, int> = 0>
  void resize(int rows, int cols) {
#endif
      /* do we need to re-allocate ? */
      if (Storage(rows, cols).num_elements() > _capacity) {
        if (m_data) delete[] m_data;
  m_data = new double[Storage(rows, cols).num_elements()];
  _capacity = Storage(rows, cols).num_elements();
} m_storage = Storage(rows, cols);
} // namespace dso

/** @brief Copy and resize (keeping the MatrixStorageType the same)
 *
 * Note that calling this function will not incurr data loss (compare with
 * the resize function), since we are resizing (re-allocating) AND copying
 * the data already stored within the data structure.
 *
 * Examples:
 * A =  +1.00  +2.00  +3.00  +4.00
 *      +5.00  +6.00  +7.00  +8.00
 *      +9.00 +10.00 +11.00 +12.00
 *     +13.00 +14.00 +15.00 +16.00
 *     +17.00 +18.00 +19.00 +20.00
 *
 *  A.cresize(10,5) =
 *      +1.00  +2.00  +3.00  +4.00  +0.00
 *      +5.00  +6.00  +7.00  +8.00  +0.00
 *      +9.00 +10.00 +11.00 +12.00  +0.00
 *     +13.00 +14.00 +15.00 +16.00  +0.00
 *     +17.00 +18.00 +19.00 +20.00  +0.00
 *      +0.00  +0.00  +0.00  +0.00  +0.00
 *      +0.00  +0.00  +0.00  +0.00  +0.00
 *      +0.00  +0.00  +0.00  +0.00  +0.00
 *      +0.00  +0.00  +0.00  +0.00  +0.00
 *      +0.00  +0.00  +0.00  +0.00  +0.00
 *
 * A.cresize(3,4) =
 *      +1.00  +2.00  +3.00  +4.00
 *      +5.00  +6.00  +7.00  +8.00
 *      +9.00 +10.00 +11.00 +12.00
 *
 * A.cresize(2,2) =
 *      +1.00  +2.00
 *      +5.00  +6.00
 *
 * Note that there is no garantee that the excess elements will be zero;
 * they can (and sometimes will) hold random values.
 */
void cresize(int rows, int cols) {
  if (rows != this->rows() || cols != this->cols()) {
    double *ptr =
        new double[StorageImplementation<S>(rows, cols).num_elements()];
    if (m_data) {
      auto pstorage = StorageImplementation<S>(rows, cols);
      int num_doubles_src;
      int num_doubles_trg;
      /* copy data (from m_data to ptr) */
      for (int s = 0;
           s < std::min(pstorage.num_slices(), m_storage.num_slices()); s++) {
        const double *__restrict__ psrc = this->slice(s, num_doubles_src);
        double *__restrict__ ptrg = ptr + pstorage.slice(s, num_doubles_trg);
        std::memcpy(ptrg, psrc,
                    sizeof(double) *
                        std::min(num_doubles_src, num_doubles_trg));
      }
      delete[] m_data;
    }
    _capacity = StorageImplementation<S>(rows, cols).num_elements();
    m_data = ptr;
    m_storage = StorageImplementation<S>(rows, cols);
  }
  /* no-op if size given is the same as the one we have */
}

template <typename T, std::enable_if_t<_is_expr_v<T>, int> = 0>
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
    reduce_copy<ReductionAssignmentOperator::EqAdd>(rhs);
  }
  return *this;
}
}
; /* class CoeffMatrix2D */

template <MatrixStorageType S>
inline void swap(CoeffMatrix2D<S> &a, CoeffMatrix2D<S> &b) noexcept {
  a.swap(b);
}

} /* namespace dso */

#endif
