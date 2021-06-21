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
    auto tag = []()->void{} // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99902
    >
  struct Tensor : Bindable<Tensor<T, Order, N, tag>>
  {
    using Bindable<Tensor<T, Order, N, tag>>::operator();

    using scalar_type = T;

    constexpr static RowMajor<Order, N> _map = {};

    DenseStorage<T, Order, N> _data;

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

    /// Make a copy of the data with a new tag, for both copy construction and
    /// assignment.
    constexpr Tensor(Tensor const&) = delete;
    constexpr auto operator=(Tensor const&) -> Tensor& = delete;

    template <auto other_tag>
    // requires (other_tag != tag) https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101155
    constexpr Tensor(Tensor<T, Order, N, other_tag> const& b)
        : _data { b._data }
    {
    }

    /// Fine to move the tag here.
    constexpr Tensor(Tensor&&) = default;
    constexpr auto operator=(Tensor&&) -> Tensor& = default;

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
  Tensor(B) -> Tensor<scalar_type_t<B>, order_v<B>, dim_v<B>>;

  /// Update the tag during a copy construction.
  template <class T, int Order, int N, auto tag>
  Tensor(Tensor<T, Order, N, tag> const&) -> Tensor<T, Order, N>;

  /// Retain the tag during a move construction.
  template <class T, int Order, int N, auto tag>
  Tensor(Tensor<T, Order, N, tag>&&) -> Tensor<T, Order, N, tag>;
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
