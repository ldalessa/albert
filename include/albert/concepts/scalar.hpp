#pragma once

#include "albert/concepts/tensor.hpp"
#include "albert/traits/is_scalar.hpp"

namespace albert::concepts
{
    // Scalars _are_ tensors of order 0.
    template <class T>
    concept scalar = concepts::tensor<T> and (std::integral<T> ||
                                              std::floating_point<T> ||
                                              traits::is_scalar<T>::value);
}
