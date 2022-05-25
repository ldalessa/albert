#pragma once

#include "albert/utils/pow.hpp"

namespace albert
{
    template <class T, int Order, int N>
    struct DenseStorage
    {
        T _data[utils::pow(N, Order)];

        constexpr static auto size()
        {
            return utils::pow(N, Order);
        }

        constexpr auto begin() const
            -> decltype(auto)
        {
            return std::begin(_data);
        }

        constexpr auto begin()
            -> decltype(auto)
        {
            return std::begin(_data);
        }

        constexpr auto end() const
            -> decltype(auto)
        {
            return std::end(_data);
        }

        constexpr auto end()
            -> decltype(auto)
        {
            return std::end(_data);
        }

        /// Normal linear access.
        constexpr auto operator[](std::integral auto i) const
            -> decltype(auto)
        {
            return _data[i];
        }

        /// Normal linear access.
        constexpr auto operator[](std::integral auto i)
            -> decltype(auto)
        {
            return _data[i];
        }
    };
}
