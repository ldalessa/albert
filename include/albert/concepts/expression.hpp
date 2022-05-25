#pragma once

#include "albert/concepts/tensor.hpp"
#include "albert/concepts/tensor_index.hpp"

namespace albert::concepts
{
    template <class T>
    concept expression = concepts::tensor<T> and requires (T t) {
        { t.outer() } -> concepts::tensor_index;
        { t.contains([]{}) } -> std::same_as<bool>;
        { t.may_alias([]{}) } -> std::same_as<bool>;
    };

}
