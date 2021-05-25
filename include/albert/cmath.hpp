#ifndef ALBERT_INCLUDE_CMATH_HPP
#define ALBERT_INCLUDE_CMATH_HPP

#include "albert/Bind.hpp"
#include "albert/traits.hpp"
#include <cmath>

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
      static_assert(rank_v<A> == 0);     // can't currently handle tensor cmath
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }

    constexpr static auto dim() -> int
    {
      return dim_v<A>;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<CMath>>) const
    {
      switch (tag) {
       case ABS:   return std::abs(a.evaluate());
       case EXP:   return std::exp(a.evaluate());
       case LOG:   return std::log(a.evaluate());
       case SQRT:  return std::sqrt(a.evaluate());
       case SIN:   return std::sin(a.evaluate());
       case COS:   return std::cos(a.evaluate());
       case TAN:   return std::tan(a.evaluate());
       case ASIN:  return std::asin(a.evaluate());
       case ACOS:  return std::acos(a.evaluate());
       case ATAN:  return std::atan(a.evaluate());
       case ATAN2: return std::atan2(a.evaluate());
       case SINH:  return std::sinh(a.evaluate());
       case COSH:  return std::cosh(a.evaluate());
       case TANH:  return std::tanh(a.evaluate());
       case ASINH: return std::asinh(a.evaluate());
       case ACOSH: return std::acosh(a.evaluate());
       case ATANH: return std::atanh(a.evaluate());
       case CEIL:  return std::ceil(a.evaluate());
       case FLOOR: return std::floor(a.evaluate());
       default:
        __builtin_abort();
      }
      __builtin_unreachable();
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
      static_assert(rank_v<A> == 0);     // can't currently handle tensor cmath
      static_assert(rank_v<B> == 0);     // can't currently handle tensor cmath
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }

    constexpr static auto dim() -> int
    {
      return dim_v<A>;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<CMath2>>) const
    {
      switch (tag) {
       case FMIN: return std::fmin(a.evaluate(), b.evaluate());
       case FMAX: return std::fmax(a.evaluate(), b.evaluate());
       case POW:  return std::pow(a.evaluate(), b.evaluate());
       default:
        __builtin_abort();
      }
      __builtin_unreachable();
    }
  };
}

#endif // ALBERT_INCLUDE_CMATH_HPP
