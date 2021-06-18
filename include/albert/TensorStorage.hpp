#ifndef ALBERT_INCLUDE_TENSOR_STORAGE_HPP
#define ALBERT_INCLUDE_TENSOR_STORAGE_HPP

#include "albert/utils.hpp"

namespace albert
{
  template <class T, int Order, int N>
  struct DenseStorage
  {
    T _data[pow(N, Order)];

    constexpr static auto size()
    {
      return pow(N, Order);
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

#endif // ALBERT_INCLUDE_TENSOR_STORAGE_HPP
