#pragma once

#include <concepts>

namespace albert::traits
{
    // Allow types other than integral and floating_point to advertise that
    // they're scalars (e.g., a rational class)
    template <class>
    struct is_scalar : std::false_type {};

    template <std::integral T>
    struct is_scalar<T> : std::true_type {};

    template <std::floating_point T>
    struct is_scalar<T> : std::true_type {};

    template <class T>
    inline constexpr auto is_scalar_v = is_scalar<std::remove_cvref_t<T>>::value;
}
