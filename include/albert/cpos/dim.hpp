#pragma once

#include "albert/concepts/scalar.hpp"
#include "albert/utils/FWD.hpp"
#include <tag_invoke/tag_invoke.hpp>

namespace albert
{
    namespace cpos
    {
        struct dim
        {
            constexpr friend auto tag_invoke(dim, concepts::scalar auto) noexcept -> int
            {
                return 0;
            }

            constexpr friend auto tag_invoke(dim, auto&& obj) noexcept -> int
                requires requires { FWD(obj).dim(); }
            {
                return FWD(obj).dim();
            }

            constexpr auto operator()(auto&& obj) const noexcept
                -> tag_invoke_result_t<dim, decltype(FWD(obj))>
            {
                return tag_invoke(*this, FWD(obj));
            }

            consteval dim(int) {}
        };
    }

    inline constexpr cpos::dim dim{0};
}
