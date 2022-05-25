#pragma once

#include "albert/concepts/tensor.hpp"
#include "albert/traits/is_scalar.hpp"
#include <concepts>

namespace albert
{
    // Scalars.
    template <class T>
    concept is_scalar = concepts::tensor<T> and (std::integral<T> ||
                                                 std::floating_point<T> ||
                                                 traits::is_scalar<T>::value);

    inline constexpr auto clang_hack = []{};

    template <class T>
    concept is_expression = concepts::tensor<T> and requires (T t) {
        { t.outer() };
        { t.contains(clang_hack) } -> std::same_as<bool>;
        { t.may_alias(clang_hack) } -> std::same_as<bool>;
    };

    template <class T>
    inline constexpr auto outer_v = std::remove_cvref_t<T>::outer();

    template <class T>
    inline constexpr int order_v = std::remove_cvref_t<T>::order();

    template <class T>
    inline constexpr int dim_v = std::remove_cvref_t<T>::dim();

    template <class T, auto tag>
    inline constexpr int contains_v = std::remove_cvref_t<T>::contains(tag);

    template <class T, auto tag>
    inline constexpr int may_alias_v = std::remove_cvref_t<T>::may_alias(tag);

    template <class T>
    inline constexpr int size_v = std::remove_cvref_t<T>::size();
}
