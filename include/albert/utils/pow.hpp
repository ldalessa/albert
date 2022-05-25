#pragma once

namespace albert::utils
{
    inline constexpr struct pow_fn
    {
        inline constexpr auto operator()(int base, int exp) const -> int
        {
            int out = 1;
            for (int i = 0; i < exp; i++) {
                out *= base;
            }
            return out;
        }

        consteval pow_fn(int) {}
    } pow{0};
}
