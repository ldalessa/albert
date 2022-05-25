#pragma once

namespace albert::concepts
{

    /// For our purposes, a tensor is anything with a order and dimension.
    template <class T>
    concept is_tensor = requires(T t)
        {
            typename scalar_type<T>::type;
            { albert::order(t) } -> std::same_as<int>;
            { albert::dim(t) } -> std::same_as<int>;
        };
}
