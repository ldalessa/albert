#pragma once

#include <type_traits>

namespace albert::traits
{
    // Allow types other than integral and floating_point to advertise that
    // they're scalars (e.g., a rational class)
    template <class>
    struct is_scalar : std::false_type {};
}
