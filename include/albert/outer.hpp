#pragma once

#include "albert/ScalarIndex.hpp"
#include "albert/concepts/scalar.hpp"
#include "albert/concepts/expression.hpp"

namespace albert
{
    struct _outer_fn
    {
        template <concepts::expresion Expr>
        constexpr auto operator()(Expr&&) const
        {
            return std::remove_cvref_t<Expr>::outer();
        }

        constexpr auto operator()(concepts::scalar auto) const
            -> TensorIndex<0>
        {
            return {};
        }

        consteval _outer_fn(int) {}
    };

    inline constexpr _outer_fn outer{0}
}
