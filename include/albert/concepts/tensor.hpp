#pragma once

#include "albert/concepts/scalar.hpp"
#include "albert/cpos/dim.hpp"
#include "albert/cpos/order.hpp"
#include "albert/traits/scalar_type.hpp"
#include <concepts>

namespace albert::concepts
{
    /// For our purposes, a tensor is anything with a order and dimension.
    template <class T>
    concept tensor = concepts::scalar<T> or requires(T t) {
        typename traits::scalar_type<T>::type;
        { albert::order(t) } -> std::same_as<int>;
        { albert::dim(t) } -> std::same_as<int>;
    };
}
