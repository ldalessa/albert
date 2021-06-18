#ifndef ALBERT_INCLUDE_GRAMMAR_HPP
#define ALBERT_INCLUDE_GRAMMAR_HPP

#include "albert/Index.hpp"
#include "albert/Tensor.hpp"
#include "albert/cmath.hpp"
#include "albert/concepts.hpp"
#include "albert/expressions.hpp"
#include "albert/utils.hpp"
#include <concepts>

namespace albert
{
  inline namespace grammar
  {
    /// Detail region defines the `promote` overloads, which are helper
    /// functions that allow the grammar to deal with non-expressions like
    /// integrals, floating point values, and unbound tensors.
    namespace detail
    {
      constexpr auto promote(is_expression auto&& a)
        -> decltype(auto)
      {
        return FWD(a);
      }

      constexpr auto promote(is_tensor auto&& a)
        -> decltype(auto)
      {
        static_assert(order_v<decltype(a)> == 0, "using tensor requires index (e.g., A(i,j)) in expression");
        return Bind(FWD(a), {}, nttp<TensorIndex<0>{}>);
      }

      constexpr auto promote(std::integral auto&& i)
        -> decltype(auto)
      {
        return Literal(i);
      }

      constexpr auto promote(std::floating_point auto&& d)
        -> decltype(auto)
      {
        return Literal(d);
      }
    }

    template <is_tensor A>
    constexpr auto operator+(A&& a)
    {
      return detail::promote(FWD(a));
    }

    template <is_tensor A, is_tensor B>
    constexpr auto operator+(A&& a, B&& b)
    {
      return Sum { detail::promote(FWD(a)), detail::promote(FWD(b)) };
    }

    template <is_expression A>
    constexpr auto operator-(A&& a)
    {
      return Negate { detail::promote(FWD(a)) };
    }

    template <is_expression A, is_expression B>
    constexpr auto operator-(A&& a, B&& b)
    {
      return Diff { detail::promote(FWD(a)), detail::promote(FWD(b)) };
    }

    template <is_expression A, is_expression B>
    constexpr auto operator*(A&& a, B&& b)
    {
      return Product { detail::promote(FWD(a)), detail::promote(FWD(b)) };
    }

    template <is_expression A, is_expression B>
    constexpr auto operator/(A&& a, B&& b)
    {
      return Product { detail::promote(FWD(a)), Inverse { detail::promote(FWD(b)) }};
    }

    template <is_expression A>
    constexpr auto D(A&& a, is_index auto i, is_index auto... is)
    {
      constexpr Index index = (i + ... + is);
      constexpr TensorIndex jindex(index);
      return Partial<decltype(detail::promote(FWD(a))), jindex> { detail::promote(FWD(a)) };
    }

    constexpr auto δ(is_index auto i, is_index auto j)
    {
      constexpr Index index = (i + j);
      constexpr TensorIndex jindex(index);
      return Delta<jindex> {};
    }

    constexpr auto ε(is_index auto i, is_index auto... is)
    {
      constexpr Index index = (i + ... + is);
      constexpr TensorIndex jindex(index);
      return Epsilon<jindex> {};
    }

    template <is_expression A>
    constexpr auto symmetrize(A&& a)
    {
      auto&& b = detail::promote(FWD(a));
      constexpr TensorIndex j = outer_v<decltype(b)>.reverse();
      return detail::promote(1) / detail::promote(2) * (b + b.template rebind<j>());
    }

    template <is_expression A>
    constexpr auto inv(A&& a)
    {
      return Inverse { detail::promote(FWD(a)) };
    }

    template <is_expression A, is_expression B>
    constexpr auto fmin(A&& a, B&& b)
    {
      assert(a.order() == 0);
      return CMath2(detail::promote(FWD(a)), detail::promote(FWD(b)), cmath_tag_v<FMIN>);
    }

    template <is_expression A, is_expression B>
    constexpr auto fmax(A&& a, B&& b)
    {
      assert(a.order() == 0);
      return CMath2(detail::promote(FWD(a)), detail::promote(FWD(b)), cmath_tag_v<FMAX>);
    }

    template <is_expression A, is_expression B>
    constexpr auto pow(A&& a, B&& b)
    {
      return CMath2(detail::promote(FWD(a)), detail::promote(FWD(b)), cmath_tag_v<POW>);
    }

    template <is_expression A>
    constexpr auto abs(A&& a)
    {
      assert(a.order() == 0);
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ABS>);
    }

    template <is_expression A>
    constexpr auto exp(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<EXP>);
    }

    template <is_expression A>
    constexpr auto log(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<LOG>);
    }

    template <is_expression A>
    constexpr auto sqrt(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<SQRT>);
    }

    template <is_expression A>
    constexpr auto sin(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<SIN>);
    }

    template <is_expression A>
    constexpr auto cos(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<COS>);
    }

    template <is_expression A>
    constexpr auto tan(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<TAN>);
    }

    template <is_expression A>
    constexpr auto asin(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ASIN>);
    }

    template <is_expression A>
    constexpr auto acos(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ACOS>);
    }

    template <is_expression A>
    constexpr auto atan(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ATAN>);
    }

    template <is_expression A>
    constexpr auto atan2(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ATAN2>);
    }

    template <is_expression A>
    constexpr auto sinh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<SINH>);
    }

    template <is_expression A>
    constexpr auto cosh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<COSH>);
    }

    template <is_expression A>
    constexpr auto tanh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<TANH>);
    }

    template <is_expression A>
    constexpr auto asinh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ASINH>);
    }

    template <is_expression A>
    constexpr auto acosh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ACOSH>);
    }

    template <is_expression A>
    constexpr auto atanh(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<ATANH>);
    }

    template <is_expression A>
    constexpr auto ceil(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<CEIL>);
    }

    template <is_expression A>
    constexpr auto floor(A&& a)
    {
      return CMath(detail::promote(FWD(a)), cmath_tag_v<FLOOR>);
    }
  } // inline namespace grammar
} // namespace albert

#endif // ALBERT_INCLUDE_GRAMMAR_HPP
