#pragma once

namespace albert::utils
{
    inline constexpr struct min_fn
    {
        inline constexpr auto operator()(int a, int b) const -> int
        {
            return (a < b) ? a : b;
        }

        consteval min_fn(int) {}
    } min{0};
}
