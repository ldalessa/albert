#pragma once

namespace albert
{
  enum CMathTag : unsigned {
    FMIN,
    FMAX,
    POW,
    ABS,
    EXP,
    LOG,
    SQRT,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    CEIL,
    FLOOR
  };

  constexpr CMathTag CMATH_TAG_MAX = CMathTag(unsigned(FLOOR) + 1);

  template <CMathTag tag>
  struct cmath_tag {};

  template <CMathTag tag>
  constexpr inline cmath_tag<tag> cmath_tag_v = {};

  template <auto tag>
  concept has_immediate = (tag < ABS);
}
