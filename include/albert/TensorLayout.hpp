#ifndef ALBERT_INCLUDE_TENSOR_LAYOUT_HPP
#define ALBERT_INCLUDE_TENSOR_LAYOUT_HPP

#include "albert/ScalarIndex.hpp"
#include <array>
#include <concepts>

namespace albert
{
  template <int Order, int N, bool RowMajor>
  struct Layout
  {
    constexpr static std::array<int, Order> stride = [] {
      std::array<int, Order> stride = {1};
      for (int i = 0; i < Order - 1; ++i) {
        stride[i + 1] = stride[i] * N;
      }
      if constexpr (RowMajor) {
        std::reverse(stride.begin(), stride.end());
      }
      return stride;
    }();

    constexpr auto operator()(ScalarIndex<Order> const& index) const -> int
    {
      int sum = 0;
      for (int i = 0; i < Order; ++i) {
        sum += index[i] * stride[i];
      }
      return sum;
    }
  };

  template <int N, bool RowMajor>
  struct Layout<0, N, RowMajor>
  {
    constexpr auto operator()(ScalarIndex<0> const&) const -> int
    {
      return 0;
    }
  };

  template <int Order, int N>
  struct RowMajor : Layout<Order, N, true>
  {
    using Layout<Order, N, true>::Layout;
  };

  template <int Order, int N>
  struct ColumnMajor : Layout<Order, N, true>
  {
    using Layout<Order, N, true>::Layout;
  };
}

#endif // ALBERT_INCLUDE_TENSOR_LAYOUT_HPP
