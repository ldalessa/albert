#pragma once

#include "albert/utils/FWD.hpp"
#include <tag_invoke/tag_invoke.hpp>

namespace albert
{
    namespace cpos
    {
        struct order
        {
            constexpr friend auto tag_invoke(order, auto&& obj) noexcept -> int
            {
                return 0;
            }

            constexpr friend auto tag_invoke(order, auto&& obj) noexcept -> int
                requires requires { FWD(obj).order(); }
            {
                return FWD(obj).order();
            }

            constexpr auto operator()(auto&& obj) const noexcept
                -> tag_invoke_result_t<order, decltype(FWD(obj))>
            {
                return tag_invoke(*this, FWD(obj));
            }

            consteval order(int) {}
        };
    }

    inline constexpr cpos::order order{0};
}
