#ifdef PRECOMPUTED_SQRT_SHFACS
#include "gravity.hpp"
#include <cmath>

dso::detail::CunninghamWeights::CunninghamWeights()
    : d1_wm1(MAX_N), d1_wm0(MAX_N), d1_wp1(MAX_N), d2_wm2(MAX_N), d2_wm1(MAX_N),
      d2_wm0(MAX_N), d2_wp1(MAX_N), d2_wp2(MAX_N) {
  for (int n = 0; n < MAX_N; ++n) {
    acc_scale[n] = std::sqrt((2.0 * n + 1.0) / (2.0 * n + 3.0));
    grad_scale[n] = std::sqrt((2.0 * n + 1.0) / (2.0 * n + 5.0));

    d1_m0_wm0[n] = std::sqrt((n + 1.0) * (n + 1.0));
    d1_m0_wp1[n] = std::sqrt((n + 1.0) * (n + 2.0) / 2.0);

    d2_m0_wm0[n] = std::sqrt((n + 1.0) * (n + 2.0) * (n + 1.0) * (n + 2.0));
    d2_m0_wp1[n] =
        std::sqrt((n + 1.0) * (n + 1.0) * (n + 2.0) * (n + 3.0) / 2.0);
    d2_m0_wp2[n] =
        std::sqrt((n + 1.0) * (n + 2.0) * (n + 3.0) * (n + 4.0) / 2.0);
  }

  for (int m = 1; m < MAX_N; ++m) {
    double *wm1 = d1_wm1.column(m);
    double *wm0 = d1_wm0.column(m);
    double *wp1 = d1_wp1.column(m);

    for (int n = m; n < MAX_N; ++n) {
      int k = n - m;
      wm1[k] = std::sqrt((n - m + 1.0) * (n - m + 2.0)) *
               ((m == 1) ? std::sqrt(2.0) : 1.0);
      wm0[k] = std::sqrt((n - m + 1.0) * (n + m + 1.0));
      wp1[k] = std::sqrt((n + m + 1.0) * (n + m + 2.0));
    }
  }

  for (int m = 2; m < MAX_N; ++m) {
    double *wm2 = d2_wm2.column(m);
    double *wm1 = d2_wm1.column(m);
    double *wm0 = d2_wm0.column(m);
    double *wp1 = d2_wp1.column(m);
    double *wp2 = d2_wp2.column(m);

    for (int n = m; n < MAX_N; ++n) {
      int k = n - m;
      wm2[k] = std::sqrt((n - m + 1.0) * (n - m + 2.0) * (n - m + 3.0) *
                         (n - m + 4.0)) *
               ((m == 2) ? std::sqrt(2.0) : 1.0);
      wm1[k] = std::sqrt((n - m + 1.0) * (n - m + 2.0) * (n - m + 3.0) *
                         (n + m + 1.0));
      wm0[k] = std::sqrt((n - m + 1.0) * (n - m + 2.0) * (n + m + 1.0) *
                         (n + m + 2.0));
      wp1[k] = std::sqrt((n - m + 1.0) * (n + m + 1.0) * (n + m + 2.0) *
                         (n + m + 3.0));
      wp2[k] = std::sqrt((n + m + 1.0) * (n + m + 2.0) * (n + m + 3.0) *
                         (n + m + 4.0));
    }
  }
}

const dso::detail::CunninghamWeights &
dso::detail::cunningham_weights() noexcept {
  static const dso::detail::CunninghamWeights w;
  return w;
}

#endif