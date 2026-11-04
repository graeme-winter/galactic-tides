#pragma once
// Tiny self-contained test helpers: no external test framework needed.

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace galactic_tides_test {
inline int &failures() {
  static int n = 0;
  return n;
}
}  // namespace galactic_tides_test

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n";   \
      ++galactic_tides_test::failures();                                                 \
    }                                                                               \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                                          \
  do {                                                                                  \
    double _a = (a), _b = (b), _t = (tol);                                              \
    if (!(std::fabs(_a - _b) <= _t)) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK_CLOSE failed: " #a " = " << _a \
                << ", " #b " = " << _b << ", tol " << _t << "\n";                       \
      ++galactic_tides_test::failures();                                                     \
    }                                                                                   \
  } while (0)

#define TEST_MAIN_RESULT()                                                    \
  (galactic_tides_test::failures() == 0                                            \
       ? (std::cout << "all checks passed\n", EXIT_SUCCESS)                   \
       : (std::cerr << galactic_tides_test::failures() << " failure(s)\n", EXIT_FAILURE))
