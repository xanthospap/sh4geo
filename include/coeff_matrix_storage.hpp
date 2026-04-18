/** @file coeff_matrix_storage.hpp
 *  @brief Compile-time storage policies for compact 2D coefficient matrices.
 *
 *  This header defines:
 *  - the enumeration @ref dso::MatrixStorageType
 *  - the primary template @ref dso::StorageImplementation
 *  - explicit specializations of @ref dso::StorageImplementation for each
 *    supported storage layout
 *
 *  The classes in this header do **not** own matrix data. They only describe:
 *  - logical dimensions
 *  - mapping from logical `(row, column)` coordinates to a flat 1D buffer
 *  - mapping from a row/column "slice" to the corresponding starting offset
 *
 *  In other words, these classes answer the question:
 *  "Given a matrix storage layout and a logical index `(i,j)`, where is the
 *  corresponding value stored in memory?"
 *
 *  This separation is useful because it isolates indexing logic from:
 *  - allocation
 *  - copying / moving
 *  - arithmetic on matrix values
 *
 *  Those concerns are handled by @ref dso::CoeffMatrix2D in another header.
 *
 *  @note
 *  All indices are zero-based.
 *
 *  @warning
 *  These classes describe the layout only. They do not check whether the
 *  caller's data pointer actually has enough storage.
 */

#ifndef __COMPACT_2D_SIMPLE_MATRIX_STORAGE_HPP__
#define __COMPACT_2D_SIMPLE_MATRIX_STORAGE_HPP__

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#ifdef DEBUG
#include <cassert>
#endif

namespace dso {

/** @brief Enumeration of supported compact matrix storage layouts.
 *
 *  The storage type determines how a logically 2D matrix is mapped into a
 *  contiguous 1D memory region.
 *
 *  ## Available layouts
 *
 *  - `RowWise`
 *    Dense matrix, stored row-by-row.
 *
 *  - `ColumnWise`
 *    Dense matrix, stored column-by-column.
 *
 *  - `Trapezoid`
 *    Lower-trapezoidal matrix, stored row-by-row.
 *    Row `r` stores `min(r+1, cols)` elements.
 *
 *  - `LwTriangularRowWise`
 *    Square lower-triangular matrix, stored row-by-row.
 *
 *  - `LwTriangularColWise`
 *    Square lower-triangular matrix, stored column-by-column.
 *
 *  @see dso::StorageImplementation
 */
enum class MatrixStorageType : char {
  RowWise,             /** Row-Wise storage */
  ColumnWise,          /** Column-Wise storage */
  Trapezoid,           /** A trapezoid matrix with row-wise storage */
  LwTriangularRowWise, /** Lower triangular, stored Row-Wise */
  LwTriangularColWise  /** Lower triangular, stored Col-Wise */
}; /* MatrixStorageType */

/** @brief Primary template for storage layout policies.
 *
 *  This template is intentionally left undefined. Only explicit
 *  specializations for supported storage layouts are provided.
 *
 *  Each specialization must provide:
 *  - dimension accessors (`nrows()`, `ncols()`)
 *  - `num_elements()`
 *  - `index(row, col)`
 *  - `slice(...)`
 *  - compile-time traits describing the storage
 *
 *  @tparam S The compile-time storage layout.
 */
template <MatrixStorageType S> class StorageImplementation {};

/** @brief Storage policy for a square lower-triangular matrix stored row-wise.
 *
 *  Only entries `(row,col)` with `col <= row` are stored.
 *
 *  ## Logical matrix shape for `N = 4`
 *
 *  @code
 *  (0,0)   .     .     .
 *  (1,0) (1,1)   .     .
 *  (2,0) (2,1) (2,2)   .
 *  (3,0) (3,1) (3,2) (3,3)
 *  @endcode
 *
 *  ## Flat contiguous storage
 *
 *  Stored row by row:
 *
 *  @code
 *  (0,0)->0
 *
 *  (1,0)->1
 *  (1,1)->2
 *
 *  (2,0)->3
 *  (2,1)->4
 *  (2,2)->5
 *
 *  (3,0)->6
 *  (3,1)->7
 *  (3,2)->8
 *  (3,3)->9
 *  @endcode
 *
 *  Therefore:
 *
 *  @code
 *  [0] = (0,0)
 *  [1] = (1,0)
 *  [2] = (1,1)
 *  [3] = (2,0)
 *  [4] = (2,1)
 *  [5] = (2,2)
 *  [6] = (3,0)
 *  [7] = (3,1)
 *  [8] = (3,2)
 *  [9] = (3,3)
 *  @endcode
 *
 *  The offset of `(row,col)` is:
 *
 *  @f[
 *    \mathrm{index}(row,col) = \frac{row(row+1)}{2} + col
 *  @f]
 *
 *  @warning
 *  Accesses with `col > row` are invalid for this storage.
 */
template <>
class StorageImplementation<MatrixStorageType::LwTriangularRowWise> {
private:
  /** num of rows (= number of columns) */
  int rows;

  /** @brief Number of stored elements in a lower-triangular square matrix.
   *  @param d Matrix dimension.
   *  @return The number of stored coefficients, i.e. `d*(d+1)/2`.
   */
  constexpr std::size_t _size(int d) const noexcept { return d * (d + 1) / 2; }

public:
  /** Basic stride/dimension */
  static constexpr const bool isRowMajor = true;
  static constexpr const bool isColMajor = false;
  static constexpr const bool isSquare = true;
  static constexpr const bool isTriangular = true;

  /** @brief Construct a lower-triangular row-wise layout.
   *  @param r Number of rows, which is also the number of columns.
   */
  constexpr StorageImplementation(int r) noexcept : rows(r) {};

  /** (Re-) set dimensions */
  void __set_dimensions(int _rows) noexcept { rows = _rows; }

  /** @brief Compute number of elements stored */
  constexpr std::size_t num_elements() const noexcept { return _size(rows); }

  /** get number of rows */
  constexpr int nrows() const noexcept { return rows; }

  /** get number of cols */
  constexpr int ncols() const noexcept { return rows; }

  /** @brief Starting offset of a row in the flat storage buffer.
   *
   *  For row `r`, all rows `0..r-1` have lengths `1,2,...,r`, so the offset is:
   *
   *  @f[
   *    \sum_{k=1}^{r} k = \frac{r(r+1)}{2}
   *  @f]
   *
   *  @param row Zero-based row index.
   *  @return Offset of the first stored element of that row.
   *
   *  @example
   *  For a `4x4` lower-triangular matrix:
   *  - `slice(0) == 0`
   *  - `slice(1) == 1`
   *  - `slice(2) == 3`
   *  - `slice(3) == 6`
   */
  constexpr int slice(int row) const noexcept { return _size(row); }

  /** @brief Index of (beggining of) row and number of elements in row.
   *
   * Return the offset from the begining of the data array, given a row
   * number. First row is row 0 (NOT row 1).
   * The input parameter \p num_elements will be set to the number of elements
   * stored in the row requested.
   */
  constexpr int slice(int row, int &num_elements) const noexcept {
    num_elements = row + 1;
    return slice(row);
  }

  /** @brief Number of slices, i.e. number of rows */
  constexpr int num_slices() const noexcept { return rows; }

  /** @brief Offset of a stored element `(row, column)`.
   *
   *  Since rows are stored contiguously and row `r` contains exactly `r+1`
   *  elements, the offset is:
   *
   *  @code
   *  slice(row) + column
   *  @endcode
   *
   *  @param row Zero-based row index.
   *  @param column Zero-based column index.
   *  @return Offset in the flat storage buffer.
   *
   *  @pre `0 <= column <= row < nrows()`
   */
  constexpr int index(int row, int column) const noexcept {
#ifdef DEBUG
    assert(row >= 0 && row < rows);
    assert(column >= 0 && column <= row);
#endif
    return slice(row) + column;
  }

  static constexpr int first_row_of_col([[maybe_unused]] int col) noexcept {
    return col;
  }

  static constexpr int first_col_of_row([[maybe_unused]] int row) noexcept {
    return 0;
  }

}; /* StorageImplementation<MatrixStorageType::LwTriangularRowWise> */

/** @brief Storage policy for a square lower-triangular matrix stored
 * column-wise.
 *
 *  Only the lower-triangular part of a logical `N x N` matrix is stored, i.e.
 *  entries `(row,col)` such that `col <= row`.
 *
 *  The unstored upper-triangular entries `(row,col)` with `col > row`
 *  do not exist in the compact storage.
 *
 *  ## Logical matrix shape
 *
 *  For `N = 4`, the logical matrix is:
 *
 *  @code
 *  (0,0)   .     .     .
 *  (1,0) (1,1)   .     .
 *  (2,0) (2,1) (2,2)   .
 *  (3,0) (3,1) (3,2) (3,3)
 *  @endcode
 *
 *  where `.` denotes an entry that is not stored.
 *
 *  ## Flat contiguous storage
 *
 *  The stored entries are written contiguously column by column:
 *
 *  @code
 *  (0,0)->0
 *  (1,0)->1
 *  (2,0)->2
 *  (3,0)->3
 *
 *  (1,1)->4
 *  (2,1)->5
 *  (3,1)->6
 *
 *  (2,2)->7
 *  (3,2)->8
 *
 *  (3,3)->9
 *  @endcode
 *
 *  Therefore the 1D memory buffer contains:
 *
 *  @code
 *  [0] = (0,0)
 *  [1] = (1,0)
 *  [2] = (2,0)
 *  [3] = (3,0)
 *  [4] = (1,1)
 *  [5] = (2,1)
 *  [6] = (3,1)
 *  [7] = (2,2)
 *  [8] = (3,2)
 *  [9] = (3,3)
 *  @endcode
 *
 *  ## Index formula
 *
 *  Column `c` starts at offset:
 *
 *  @f[
 *    \mathrm{slice}(c) = cN - \frac{c(c-1)}{2}
 *  @f]
 *
 *  and `(row,col)` is located at:
 *
 *  @f[
 *    \mathrm{index}(row,col) = \mathrm{slice}(col) + (row-col)
 *  @f]
 *
 *  @note
 *  This layout is efficient when full stored columns are traversed frequently,
 *  because each stored column is contiguous in memory.
 *
 *  @warning
 *  Accesses with `col > row` are invalid for this storage.
 */
template <>
class StorageImplementation<MatrixStorageType::LwTriangularColWise> {
private:
  /** num of rows (= number of columns) */
  int rows;

public:
  /** Basic stride/dimension */
  static constexpr const bool isRowMajor = false;
  static constexpr const bool isColMajor = true;
  static constexpr const bool isSquare = true;
  static constexpr const bool isTriangular = true;

  /** Constructor; not interested in number of cols */
  constexpr StorageImplementation(int r) noexcept : rows(r) {}

  /** @brief Compute number of elements stored */
  constexpr std::size_t num_elements() const noexcept {
    return rows * (rows + 1) / 2;
  }

  /** (Re-) set dimensions */
  void __set_dimensions(int _rows) noexcept { rows = _rows; }

  /** number of rows */
  constexpr int nrows() const noexcept { return rows; }

  /** number of columns */
  constexpr int ncols() const noexcept { return rows; }

  /** @brief Starting offset of a column in the flat storage buffer.
   *
   *  Column `c` contains `rows - c` stored elements.
   *  The offset is the total length of all previous stored columns.
   *
   *  For a matrix with dimension `N`, the offset of column `c` is:
   *
   *  @f[
   *    cN - \frac{c(c-1)}{2}
   *  @f]
   *
   *  @param col Zero-based column index.
   *  @return Offset of the first stored element of that column.
   *
   *  @example
   *  For `N = 4`:
   *  - `slice(0) == 0`
   *  - `slice(1) == 4`
   *  - `slice(2) == 7`
   *  - `slice(3) == 9`
   */
  constexpr int slice(int col) const noexcept {
    return col * rows - col * (col - 1) / 2;
  }

  constexpr int slice(int col, int &num_elements) const noexcept {
    num_elements = nrows() - col;
    return slice(col);
  }

  /** @brief Number of slices, i.e. number of cols */
  constexpr int num_slices() const noexcept { return rows; }

  /** @brief Offset of a stored element `(row, column)`.
   *
   *  In a column-wise lower-triangular layout, column `c` starts at `slice(c)`
   *  and the element `(row,c)` is the `(row-c)`-th entry inside that column.
   *
   *  Therefore:
   *
   *  @code
   *  index(row, col) = slice(col) + (row - col)
   *  @endcode
   *
   *  @param row Zero-based row index.
   *  @param col Zero-based column index.
   *  @return Offset in the flat storage buffer.
   *
   *  @pre `0 <= col <= row < nrows()`
   */
  constexpr int index(int row, int col) const noexcept {
#ifdef DEBUG
    assert(row >= 0 && row < rows);
    assert(col >= 0 && col <= row);
#endif
    return slice(col) + (row - col);
  }

  static constexpr int first_row_of_col([[maybe_unused]] int col) noexcept {
    return col;
  }
  static constexpr int first_col_of_row([[maybe_unused]] int row) noexcept {
    return 0;
  }
}; /* StorageImplementation<MatrixStorageType::LwTriangularColWise> */

/** @brief Storage policy for a lower-trapezoidal matrix stored row-wise.
 *
 *  This layout stores the first `min(row+1, cols)` elements of each row.
 *
 *  It can be viewed as a lower-triangular layout truncated to a maximum
 *  number of columns.
 *
 *  The number of stored elements in row `r` is:
 *
 *  @code
 *  pts_in_row(r) = min(r + 1, cols)
 *  @endcode
 *
 *  This is useful for coefficient tables where the logical number of rows
 *  (degree) exceeds the logical number of columns (order).
 *
 *  ## Logical matrix shape for `rows=5`, `cols=3`
 *
 *  @code
 *  (0,0)   .     .
 *  (1,0) (1,1)   .
 *  (2,0) (2,1) (2,2)
 *  (3,0) (3,1) (3,2)
 *  (4,0) (4,1) (4,2)
 *  @endcode
 *
 *  ## Flat contiguous storage
 *
 *  @code
 *  (0,0)->0
 *
 *  (1,0)->1
 *  (1,1)->2
 *
 *  (2,0)->3
 *  (2,1)->4
 *  (2,2)->5
 *
 *  (3,0)->6
 *  (3,1)->7
 *  (3,2)->8
 *
 *  (4,0)->9
 *  (4,1)->10
 *  (4,2)->11
 *  @endcode
 *
 *  Therefore:
 *
 *  @code
 *  [0]  = (0,0)
 *  [1]  = (1,0)
 *  [2]  = (1,1)
 *  [3]  = (2,0)
 *  [4]  = (2,1)
 *  [5]  = (2,2)
 *  [6]  = (3,0)
 *  [7]  = (3,1)
 *  [8]  = (3,2)
 *  [9]  = (4,0)
 *  [10] = (4,1)
 *  [11] = (4,2)
 *  @endcode
 */
template <> class StorageImplementation<MatrixStorageType::Trapezoid> {
private:
  /** number of rows */
  int rows;
  /** number of columns */
  int cols;

public:
  /** Basic stride/dimension */
  static constexpr const bool isRowMajor = true;
  static constexpr const bool isColMajor = false;
  static constexpr const bool isSquare = false;
  static constexpr const bool isTriangular = true;

  /** Constructor given number of rows and num of columns */
  constexpr StorageImplementation(int r, int c) noexcept : rows(r), cols(c) {};

  /** number of rows */
  constexpr int nrows() const noexcept { return rows; }

  /** number of columns */
  constexpr int ncols() const noexcept { return cols; }

  /** (Re-) set dimensions */
  void __set_dimensions(int _rows, int _cols) noexcept {
    rows = _rows;
    cols = _cols;
  }

  /** @brief Total number of stored elements.
   *
   *  There are two regimes:
   *
   *  - If `rows <= cols`, the shape is purely triangular and:
   *    @f[
   *      \frac{rows(rows+1)}{2}
   *    @f]
   *
   *  - If `rows > cols`, the first `cols` rows grow triangularly, and all later
   *    rows contain exactly `cols` elements:
   *    @f[
   *      \frac{cols(cols+1)}{2} + (rows-cols)\,cols
   *    @f]
   */
  constexpr std::size_t num_elements() const noexcept {
    if (rows <= cols)
      return rows * (rows + 1) / 2;
    return cols * (cols + 1) / 2 + (rows - cols) * cols;
  }

  /** @brief Number of elements (data points) stored for a given row */
  constexpr int pts_in_row(int row) const noexcept {
    return std::min(row + 1, cols);
  }

  /** @brief Starting offset of a row in the flat storage buffer.
   *
   *  For `row < cols`, the storage is still in the triangular-growth region:
   *
   *  @f[
   *    \frac{row(row+1)}{2}
   *  @f]
   *
   *  For `row >= cols`, all previous rows beyond the triangular core contribute
   *  exactly `cols` elements:
   *
   *  @f[
   *    \frac{cols(cols+1)}{2} + (row-cols)\,cols
   *  @f]
   *
   *  @param row Zero-based row index.
   *  @return Offset of the first stored element of that row.
   */
  constexpr int slice(int row) const noexcept {
    if (row < cols)
      return row * (row + 1) / 2;
    return cols * (cols + 1) / 2 + (row - cols) * cols;
  }

  constexpr int slice(int row, int &num_elements) const noexcept {
    num_elements = pts_in_row(row);
    return slice(row);
  }

  /** @brief Number of slices, i.e. number of rows */
  constexpr int num_slices() const noexcept { return rows; }

  /** @brief Index of element (row, column) in the data array.
   * E.g. data[element_offset(1,2)] will return the element in the second row,
   *  and third column.
   */
  constexpr int index(int row, int column) const noexcept {
#ifdef DEBUG
    assert(row >= 0 && row < rows);
    assert(column >= 0 && column < pts_in_row(row));
#endif
    return slice(row) + column;
  }

  static constexpr int first_row_of_col([[maybe_unused]] int col) noexcept {
    return col;
  }
  static constexpr int first_col_of_row([[maybe_unused]] int row) noexcept {
    return 0;
  }

}; /* StorageImplementation<MatrixStorageType::Trapezoid> */

/** @brief Storage policy for a dense row-major matrix.
 *
 *  A matrix with `rows` rows and `cols` columns is stored in the familiar
 *  row-major order:
 *
 *  @code
 *  (0,0),(0,1),...,(0,cols-1),
 *  (1,0),(1,1),...,(1,cols-1),
 *  ...
 *  @endcode
 *
 *  For `rows=3`, `cols=4`:
 *
 *  ## Logical matrix
 *  @code
 *  (0,0) (0,1) (0,2) (0,3)
 *  (1,0) (1,1) (1,2) (1,3)
 *  (2,0) (2,1) (2,2) (2,3)
 *  @endcode
 *
 *  ## Flat contiguous storage
 *  @code
 *  (0,0)->0   (0,1)->1   (0,2)->2   (0,3)->3
 *  (1,0)->4   (1,1)->5   (1,2)->6   (1,3)->7
 *  (2,0)->8   (2,1)->9   (2,2)->10  (2,3)->11
 *  @endcode
 *
 *  The offset formula is:
 *
 *  @code
 *  index(row, col) = row * cols + col
 *  @endcode
 *
 *  This layout is efficient when rows are traversed contiguously.
 */
template <> struct StorageImplementation<MatrixStorageType::RowWise> {
private:
  /** number of rows */
  int rows;
  /** number of columns */
  int cols;

public:
  /** Basic stride/dimension */
  static constexpr const bool isRowMajor = true;
  static constexpr const bool isColMajor = false;
  static constexpr const bool isSquare = false;
  static constexpr const bool isTriangular = false;

  /** Constructor given number of rows and number of columns */
  constexpr StorageImplementation(int r, int c) noexcept : rows(r), cols(c) {};

  /** get number of rows */
  constexpr int nrows() const noexcept { return rows; }

  /* get number of columns */
  constexpr int ncols() const noexcept { return cols; }

  /** (Re-) set dimensions */
  void __set_dimensions(int _rows, int _cols) noexcept {
    rows = _rows;
    cols = _cols;
  }

  /** @brief Number of elements in matrix */
  constexpr std::size_t num_elements() const noexcept { return rows * cols; }

  /** @brief Index of element (row, column) in the data array.
   *  E.g. data[element_offset(1,2)] will return the element in the second
   *  row, and third column.
   */
  constexpr int index(int row, int column) const noexcept {
    return row * cols + column;
  }

  /** @brief Index/offset of given row.
   *
   * Return the offset from the begining of the data array, given a row
   * number.
   * First row is row 0 (NOT row 1).
   * That means that if the data is stored in an array e.g.
   *   double *data = new double[num_pts];
   *   double *row_3 = data[0] + slice(2);
   * will point to the first (0) element of the third row.
   */
  constexpr int slice(int row) const noexcept { return row * cols; }

  constexpr int slice(int row, int &num_elements) const noexcept {
    num_elements = ncols();
    return slice(row);
  }

  /** @brief Number of slices, i.e. number of rows */
  constexpr int num_slices() const noexcept { return rows; }

  static constexpr int first_row_of_col([[maybe_unused]] int col) noexcept {
    return 0;
  }
  static constexpr int first_col_of_row([[maybe_unused]] int row) noexcept {
    return 0;
  }
}; /* StorageImplementation<MatrixStorageType::RowWise> */

/** @brief Storage policy for a dense column-major matrix.
 *
 *  A matrix with `rows` rows and `cols` columns is stored column-by-column:
 *
 *  @code
 *  (0,0),(1,0),...,(rows-1,0),
 *  (0,1),(1,1),...,(rows-1,1),
 *  ...
 *  @endcode
 *
 *  For `rows=3`, `cols=4`:
 *
 *  ## Logical matrix
 *  @code
 *  (0,0) (0,1) (0,2) (0,3)
 *  (1,0) (1,1) (1,2) (1,3)
 *  (2,0) (2,1) (2,2) (2,3)
 *  @endcode
 *
 *  ## Flat contiguous storage
 *  @code
 *  (0,0)->0   (1,0)->1   (2,0)->2
 *  (0,1)->3   (1,1)->4   (2,1)->5
 *  (0,2)->6   (1,2)->7   (2,2)->8
 *  (0,3)->9   (1,3)->10  (2,3)->11
 *  @endcode
 *
 *  The offset formula is:
 *
 *  @code
 *  index(row, col) = col * rows + row
 *  @endcode
 *
 *  This layout is efficient when columns are traversed contiguously.
 */
template <> class StorageImplementation<MatrixStorageType::ColumnWise> {
private:
  /** number of rows */
  int rows;
  /** number of columns */
  int cols;

public:
  /** Basic stride/dimension */
  static constexpr const bool isRowMajor = false;
  static constexpr const bool isColMajor = true;
  static constexpr const bool isSquare = false;
  static constexpr const bool isTriangular = false;

  /** Constructor given number of rows and number of columns */
  constexpr StorageImplementation(int r, int c) noexcept : rows(r), cols(c) {};

  /** get number of rows */
  constexpr int nrows() const noexcept { return rows; }

  /** get number of columns */
  constexpr int ncols() const noexcept { return cols; }

  /** (Re-) set dimensions */
  void __set_dimensions(int _rows, int _cols) noexcept {
    rows = _rows;
    cols = _cols;
  }

  /** @brief Number of elements in matrix */
  constexpr std::size_t num_elements() const noexcept { return rows * cols; }

  /** @brief Index of element (row, column) in the data array.
   *
   *  E.g. data[element_offset(1,2)] will return the element in the second
   *  row, and third column.
   */
  constexpr int index(int row, int column) const noexcept {
    return column * rows + row;
  }

  /** @brief Index/offset of given column.
   *
   * Return the offset from the begining of the data array, given a column
   * number.
   * First column is column 0 (NOT row 1).
   * That means that if the data is stored in an array e.g.
   *   double *data = new double[num_pts];
   *   double *col_3 = data[0] + slice(2);
   * will point to the first (0) element of the third column.
   */
  constexpr int slice(int col) const noexcept { return col * rows; }

  constexpr int slice(int col, int &num_elements) const noexcept {
    num_elements = nrows();
    return slice(col);
  }

  constexpr int num_slices() const noexcept { return ncols(); }

  static constexpr int first_row_of_col([[maybe_unused]] int col) noexcept {
    return 0;
  }
  static constexpr int first_col_of_row([[maybe_unused]] int row) noexcept {
    return 0;
  }

}; /* StorageImplementation<MatrixStorageType::ColumnWise> */

} /* namespace dso */

#endif
