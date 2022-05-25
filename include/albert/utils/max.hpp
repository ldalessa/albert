#pragma once

namespace albert::utils
{
    inline constexpr struct max_fn
    {
        inline constexpr auto operator()(int a, int b) const -> int
        {
            return (a < b) ? b : a;
        }

        consteval max_fn(int) {}
    } max{0};
}
