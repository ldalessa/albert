#ifndef ALBERT_INCLUDE_ALBERT_CPOS_HPP
#define ALBERT_INCLUDE_ALBERT_CPOS_HPP

#include "albert/utils.hpp"
#include <tag_invoke/tag_invoke.hpp>

namespace albert
{
  inline namespace cpos
  {
    constexpr inline struct outer_tag
    {
      constexpr friend auto tag_invoke(outer_tag, auto&& obj) noexcept
        requires requires { FWD(obj).outer(); }
      {
        return FWD(obj).outer();
      }

      constexpr auto operator()(auto&& obj) const noexcept
        -> tag_invoke_result_t<outer_tag, decltype(FWD(obj))>
      {
        return tag_invoke(*this, FWD(obj));
      }
    } outer;

    constexpr inline struct order_tag
    {
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

    constexpr inline struct contains_tag
    {
      constexpr friend auto tag_invoke(contains_tag, auto&& obj, auto&& tag) noexcept
        -> bool
        requires requires { FWD(obj).contains(FWD(tag)); }
      {
        return FWD(obj).contains(FWD(tag));
      }

      constexpr auto operator()(auto&& obj, auto&& tag) const noexcept
        -> bool
      {
        return tag_invoke(*this, FWD(obj), FWD(tag));
      }
    } contains;

    constexpr inline struct may_alias_tag
    {
      constexpr friend auto tag_invoke(may_alias_tag, auto&& obj, auto&& tag) noexcept
        -> bool
        requires requires { FWD(obj).may_alias(FWD(tag)); }
      {
        return FWD(obj).may_alias(FWD(tag));
      }

      constexpr auto operator()(auto&& obj, auto&& tag) const noexcept
        -> bool
      {
        return tag_invoke(*this, FWD(obj), FWD(tag));
      }
    } may_alias;
  }
}

#endif// ALBERT_INCLUDE_ALBERT_CPOS_HPP
