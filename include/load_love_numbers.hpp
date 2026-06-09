#ifndef __DSO_LOAD_LOVE_NUMBERS_LIST_HPP__
#define __DSO_LOAD_LOVE_NUMBERS_LIST_HPP__

#include <array>

namespace dso {
struct LoadLoveNumbers {
  static constexpr int NUM = 500;
  std::array<double, NUM> h, l, k;
}; /* struct LoadLoveNumbers */

extern LoadLoveNumbers groopsLoadLoveNumbers;

} /* namespace dso */
#endif