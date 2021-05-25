#ifndef ALBERT_INCLUDE_ROW_MAJOR_HPP
#define ALBERT_INCLUDE_ROW_MAJOR_HPP

#include "albert/ScalarIndex.hpp"
#include <array>
#include <concepts>

namespace albert
{
  template <int Rank, int N>
  struct RowMajor
  {
    constexpr static std::array<int, Rank> stride = [] {
      int stride[Rank];
      stride[Rank - 1] = 1;
      for (int i = Rank - 1; i > 0; --i) {
        stride[i - 1] = stride[i] * N;
      }
      return std::to_array(stride);
    }();

    constexpr auto operator()(std::integral auto... is) const -> int
      requires(sizeof...(is) == Rank)
    {
      int sum = 0;
      int i = 0;
      ((sum += (is * stride[i++])), ...);
      return sum;
    }

    constexpr auto operator()(ScalarIndex<Rank> const& index) const -> int
    {
      int sum = 0;
      for (int i = 0; i < Rank; ++i) {
        sum += index[i] * stride[i];
      }
      return sum;
    }
  };

  template <>
  struct RowMajor<0, 0>
  {
    constexpr auto operator()() const -> int
    {
      return 0;
    }

    constexpr auto operator()(ScalarIndex<0> const&) const -> int
    {
      return 0;
    }
  };
}

#endif // ALBERT_INCLUDE_ROW_MAJOR_HPP
