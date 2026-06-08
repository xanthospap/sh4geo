#include "gravity.hpp"
#include <array>
#include <cmath>

dso::NormalizedLegendreFactors::NormalizedLegendreFactors() noexcept
    : f1(MAX_SIZE_FOR_ALF_FACTORS), f2(MAX_SIZE_FOR_ALF_FACTORS) {
  constexpr const int N = MAX_SIZE_FOR_ALF_FACTORS;
  f1.fill_with(0e0);
  f2.fill_with(0e0);

  /* factors for the recursion ((2n+1)/2n)^(1/2) */
  f1(1, 1) = std::sqrt(3e0);
  for (int n = 2; n < N; n++) {
    f1(n, n) = std::sqrt((2e0 * n + 1e0) / (2e0 * n));
  }

  /* factors for the recursion */
  for (int m = 0; m < N - 1; m++) {
    for (int n = m + 1; n < N; n++) {
      const double f = (2e0 * n + 1e0) / static_cast<double>((n + m) * (n - m));
      /* f1_nm = B_nm */
      f1(n, m) = std::sqrt(f * (2e0 * n - 1e0));
      /* f2_nm = B_nm / Bn-1m */
      f2(n, m) =
          -std::sqrt(f * (n - m - 1e0) * (n + m - 1e0) / (2e0 * n - 3e0));
    }
  }

  /* factors for acceleration */
  for (int n = 0; n < N; n++)
    f3[n] = std::sqrt((double)(2 * n + 1) / (2 * n + 3));

  /* all computations done */
  return;
}