#pragma once

#include "albert/traits/is_scalar.hpp"
#include "albert/concepts/tensor.hpp"

namespace albert::concepts
{
    template <class T>
    concept scalar = tensor<T> and traits::is_scalar_v<T>;
}
