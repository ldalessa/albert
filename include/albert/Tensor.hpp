#ifndef ALBERT_INCLUDE_TENSOR_HPP
#define ALBERT_INCLUDE_TENSOR_HPP

#include "albert/Bind.hpp"
#include "albert/TensorStorage.hpp"
#include "albert/TensorLayout.hpp"
#include "albert/concepts.hpp"
#include "albert/evaluate.hpp"
#include "albert/utils.hpp"

namespace albert
{
  template <
    class T,
    int Order,
    int N,
    template <int, int> class Layout = RowMajor,
    template <class, int, int> class Storage = DenseStorage
    >
  struct Tensor : Bindable<Tensor<T, Order, N, Layout, Storage>>
  {
    using Bindable<Tensor<T, Order, N, Layout, Storage>>::operator();

    using scalar_type = T;

    constexpr static Layout<Order, N> _map = {};

    Storage<T, Order, N> _data;

    constexpr operator scalar_type() const requires(Order == 0)
    {
      return _data[0];
    }

    constexpr static auto size()
      -> int
    {
      return pow(N, Order);
    }

    constexpr static auto order()
      -> int
    {
      return Order;
    }

    constexpr static auto dim()
      -> int
    {
      return N;
    }

    constexpr Tensor() = default;

    constexpr Tensor(std::convertible_to<T> auto t, std::convertible_to<T> auto... ts)
      : _data { static_cast<T>(t), static_cast<T>(ts)... }
    {
      static_assert(sizeof...(ts) < size());
    }

    /// Construct a tensor from an expression.
    template <is_expression B>
    constexpr Tensor(B&& b)
    {
      static_assert(order_v<B> == Order, "expression order does not match");
      Bind(*this, {}, nttp<outer_v<B>>) = FWD(b);
    }

    template <is_expression B>
    constexpr auto operator=(B&& b) &
      -> decltype(auto)
    {
      static_assert(order_v<B> == Order, "expression order does not match");
      return std::move(Bind(*this, {}, nttp<outer_v<B>>) = FWD(b));
    }

    template <is_expression B>
    constexpr auto operator=(B&& b) &&
      -> decltype(auto)
    {
      static_assert(order_v<B> == Order, "expression order does not match");
      return std::move(Bind(std::move(*this), {}, nttp<outer_v<B>>) = FWD(b));
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

    /// Multidimensional indexing via aggregate.
    constexpr auto evaluate(ScalarIndex<Order> const& index) const
      -> decltype(auto)
    {
      int i = _map(index);
      return _data[i];
    }

    /// Multidimensional indexing via aggregate.
    constexpr auto evaluate(ScalarIndex<Order> const& index)
      -> decltype(auto)
    {
      int i = _map(index);
      return _data[i];
    }
  };

  /// Infer a tensor type for an expression.
  template <is_expression B>
  Tensor(B) -> Tensor<scalar_type_t<B>, order_v<B>, dim_v<B>, RowMajor, DenseStorage>;
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
