#pragma once

#include <concepts>

namespace albert::concepts
{
    template <class T>
    concept index = std::integral<T> or requires {
        typename std::remove_cvref_t<T>::index_tag;
    };
}
