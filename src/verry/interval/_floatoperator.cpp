#include <pybind11/pybind11.h>

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <utility>

#ifdef _MSC_VER
#define FENV_ACCESS_ON _Pragma("fenv_access(on)")
#define FP_CONTRACT_OFF _Pragma("fp_contract(off)")
#else
#define FENV_ACCESS_ON _Pragma("STDC FENV_ACCESS ON")
#define FP_CONTRACT_OFF _Pragma("STDC FP_CONTRACT OFF")
#endif

static_assert(FLT_EVAL_METHOD == 0);
static_assert(std::numeric_limits<double>::is_iec559);

double cadd(double lhs, double rhs) {
  FENV_ACCESS_ON
  FP_CONTRACT_OFF
  volatile double x = lhs;
  volatile double y = rhs;
  std::fesetround(FE_UPWARD);
  volatile double z = x + y;
  std::fesetround(FE_TONEAREST);
  return z;
}

double cmul(double lhs, double rhs) {
  FENV_ACCESS_ON
  FP_CONTRACT_OFF
  volatile double x = lhs;
  volatile double y = rhs;
  std::fesetround(FE_UPWARD);
  volatile double z = x * y;
  std::fesetround(FE_TONEAREST);
  return z;
}

double cdiv(double lhs, double rhs) {
  FENV_ACCESS_ON
  FP_CONTRACT_OFF
  volatile double x = lhs;
  volatile double y = rhs;
  std::fesetround(FE_UPWARD);
  volatile double z = x / y;
  std::fesetround(FE_TONEAREST);
  return z;
}

double csqr(double value) {
  FENV_ACCESS_ON
  FP_CONTRACT_OFF
  volatile double x = value;
  std::fesetround(FE_UPWARD);
  volatile double y = std::sqrt(x);
  std::fesetround(FE_TONEAREST);
  return y;
}

double fsqr(double value) {
  FENV_ACCESS_ON
  FP_CONTRACT_OFF
  volatile double x = value;
  std::fesetround(FE_DOWNWARD);
  volatile double y = std::sqrt(x);
  std::fesetround(FE_TONEAREST);
  return y;
}

PYBIND11_MODULE(_floatoperator, mod) {
  mod.def("cadd", &cadd);
  mod.def("cmul", &cmul);
  mod.def("cdiv", &cdiv);
  mod.def("csqr", &csqr);
  mod.def("fsqr", &fsqr);
}
