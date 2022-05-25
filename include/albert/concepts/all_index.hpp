#pragma once

#include "albert/concepts/index.hpp"

namespace albert::concepts
{
    template<typename... Ts>
    concept all_index = (concepts::index<Ts> && ...);
}
