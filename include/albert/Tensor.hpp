#ifndef ALBERT_INCLUDE_TENSOR_HPP
#define ALBERT_INCLUDE_TENSOR_HPP

#include "albert/Bind.hpp"

namespace albert
{
  template <class T, std::size_t Rank, std::size_t N>
  struct Tensor : Bindable<Tensor<T, Rank, N>>
  {
    using tensor_tag = void;

    constexpr static std::size_t rank()
    {
      return Rank;
    }
  };
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
