#pragma once

#include "albert/concepts/index.hpp"

namespace albert::concepts
{
    template<typename... Ts>
    concept all_integral_index = all_index<Ts...> and (std::integral<Ts> && ...);
}
