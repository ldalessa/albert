#pragma once

#include "albert/traits/is_scalar.hpp"
#include <concepts>

namespace albert::traits
{
    // Extract the scalar type of some expression type.
    template <class T>
    struct scalar_type;

    template <class T>
    using scalar_type_t = typename scalar_type<T>::type;

    // Pure scalar types are their own scalar types.
    template <class T>
    requires (is_scalar_v<T>)
    struct scalar_type<T>
    {
        using type = T;
    };

    // Classes with a dependent scalar_type are handled.
    template <class T>
    requires requires { typename std::remove_cvref_t<T>::scalar_type; }
    struct scalar_type<T>
    {
        using type = typename std::remove_cvref_t<T>::scalar_type;
    };
}
