#pragma once

#include <type_traits>

namespace albert::concepts
{
    template <class T>
    concept tensor_index = requires {
        typename std::remove_cvref_t<T>::tensor_index_tag;
    };
}
