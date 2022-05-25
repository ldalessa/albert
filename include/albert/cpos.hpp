#ifndef ALBERT_INCLUDE_ALBERT_CPOS_HPP
#define ALBERT_INCLUDE_ALBERT_CPOS_HPP

#include "albert/utils/FWD.hpp"
#include <tag_invoke/tag_invoke.hpp>

namespace albert
{
  inline namespace cpos
  {
    constexpr inline struct order_tag
    {
      constexpr friend auto tag_invoke(order_tag, auto&&) noexcept -> int
      {
        return 0;
      }

      constexpr friend auto tag_invoke(order_tag, auto&& obj) noexcept -> int
        requires requires { FWD(obj).order(); }
      {
        return FWD(obj).order();
      }

      constexpr auto operator()(auto&& obj) const noexcept
        -> tag_invoke_result_t<order_tag, decltype(FWD(obj))>
      {
        return tag_invoke(*this, FWD(obj));
      }
    } order;

    constexpr inline struct dim_tag
    {
      constexpr friend auto tag_invoke(dim_tag, auto&& obj) noexcept -> int
      {
        return 0;
      }

      constexpr friend auto tag_invoke(dim_tag, auto&& obj) noexcept -> int
        requires requires { FWD(obj).dim(); }
      {
        return FWD(obj).dim();
      }

      constexpr auto operator()(auto&& obj) const noexcept
        -> tag_invoke_result_t<dim_tag, decltype(FWD(obj))>
      {
        return tag_invoke(*this, FWD(obj));
      }
    } dim;
  }
}

#endif// ALBERT_INCLUDE_ALBERT_CPOS_HPP
