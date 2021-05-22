#ifndef ALBERT_INCLUDE_CMATH_HPP
#define ALBERT_INCLUDE_CMATH_HPP

#include "albert/Bind.hpp"
#include "albert/traits.hpp"

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

  template <class A, CMathTag tag>
  struct CMath : Bindable<CMath<A, tag>>
  {
    static_assert(POW < tag and tag < CMATH_TAG_MAX);
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr CMath(A a, cmath_tag<tag>)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };

  template <class A, class B, CMathTag tag>
  struct CMath2 : Bindable<CMath<A, tag>>
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

    A a;
    B b;

    constexpr CMath2(A a, B b, cmath_tag<tag>)
        : a(std::move(a))
        , b(std::move(b))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };
}

#endif // ALBERT_INCLUDE_CMATH_HPP
