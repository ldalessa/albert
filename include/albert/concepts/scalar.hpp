#pragma once

#include "albert/traits/is_scalar.hpp"

namespace albert::concepts
{
    template <class T>
    concept scalar = (
            std::integral<std::remove_cvref_t<T>> ||
            std::floating_point<std::remove_cvref_t<T>> ||
            traits::is_scalar<T>::value);
}
