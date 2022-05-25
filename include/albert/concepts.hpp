#pragma once

#include <concepts>

namespace albert
{
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
