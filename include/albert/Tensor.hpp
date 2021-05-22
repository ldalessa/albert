#ifndef ALBERT_INCLUDE_TENSOR_HPP
#define ALBERT_INCLUDE_TENSOR_HPP

#include "albert/Bind.hpp"
#include "albert/concepts.hpp"
#include "albert/traits.hpp"

namespace albert
{
  template <class T, std::size_t Rank, std::size_t N>
  struct Tensor : Bindable<Tensor<T, Rank, N>>
  {
    using tensor_tag = void;

    constexpr Tensor() = default;

    template <is_tree B>
    constexpr Tensor(B&& b)
    {
      static_assert(rank_v<B> == Rank);
    }

    template <is_tree B>
    constexpr Tensor& operator=(B&& b)
    {
      assert(rank_v<B> == Rank);
      return *this;
    }

    constexpr static std::size_t rank()
    {
      return Rank;
    }
  };
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
