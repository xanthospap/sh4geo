#include "gravity"
#include <array>
#include <cmath>

dso::detail::PrecomputedShSqrts::PrecomputedShSqrts() noexcept {
  for (int i = 0; i < N; i++) {
    sqnp3[i] = std::sqrt((double)(2 * i + 1) / (2 * i + 3));
    sqnp5[i] = std::sqrt((double)(2 * i + 1) / (2 * i + 5));
  }
}

const dso::detail::PrecomputedShSqrts &
dso::detail::precomputed_sh_sqrts() noexcept {
  static const dso::detail::PrecomputedShSqrts p;
  return p;
}