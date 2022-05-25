#pragma once

namespace albert::utils
{
    template <auto...>
    struct nttp_args {};

    template <auto... args>
    inline constexpr nttp_args<args...> nttp = {};
}
