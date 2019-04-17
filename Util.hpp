#include <cmath>
#include <functional>
#include "Matrix.hpp"

#ifndef UTIL_H
#define UTIL_H

namespace Util {
function<double(double)> sigmoid = [](double x) {
  return 1.0 / (1.0 + exp(-x));
};

function<double(double)> sigmoid_derivative = [](double x) {
  return x * (1.0 - x);
};
}  // namespace Util

#endif