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

    constexpr auto evaluate(ScalarIndex<rank_v<CMath>> const&) const
    {
      using std::abs;
      using std::exp;
      using std::log;
      using std::sqrt;
      using std::sin;
      using std::cos;
      using std::tan;
      using std::asin;
      using std::acos;
      using std::atan;
      using std::atan2;
      using std::sinh;
      using std::cosh;
      using std::tanh;
      using std::asinh;
      using std::acosh;
      using std::atanh;
      using std::ceil;
      using std::floor;

      switch (tag) {
       case ABS:   return abs(a.evaluate());
       case EXP:   return exp(a.evaluate());
       case LOG:   return log(a.evaluate());
       case SQRT:  return sqrt(a.evaluate());
       case SIN:   return sin(a.evaluate());
       case COS:   return cos(a.evaluate());
       case TAN:   return tan(a.evaluate());
       case ASIN:  return asin(a.evaluate());
       case ACOS:  return acos(a.evaluate());
       case ATAN:  return atan(a.evaluate());
       case ATAN2: return atan2(a.evaluate());
       case SINH:  return sinh(a.evaluate());
       case COSH:  return cosh(a.evaluate());
       case TANH:  return tanh(a.evaluate());
       case ASINH: return asinh(a.evaluate());
       case ACOSH: return acosh(a.evaluate());
       case ATANH: return atanh(a.evaluate());
       case CEIL:  return ceil(a.evaluate());
       case FLOOR: return floor(a.evaluate());
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

    constexpr auto evaluate(ScalarIndex<rank_v<CMath2>> const&) const
    {
      using std::fmin;
      using std::fmax;
      using std::pow;
      switch (tag) {
       case FMIN: return fmin(a.evaluate(), b.evaluate());
       case FMAX: return fmax(a.evaluate(), b.evaluate());
       case POW:  return pow(a.evaluate(), b.evaluate());
       default:
        __builtin_abort();
      }
      __builtin_unreachable();
    }
  };
}

#endif // ALBERT_INCLUDE_CMATH_HPP
