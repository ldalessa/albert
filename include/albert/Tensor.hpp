#ifndef ALBERT_INCLUDE_TENSOR_HPP
#define ALBERT_INCLUDE_TENSOR_HPP

#include "albert/Bind.hpp"
#include "albert/RowMajor.hpp"
#include "albert/concepts.hpp"
#include "albert/evaluate.hpp"
#include "albert/traits.hpp"
#include "albert/utils.hpp"

namespace albert
{
  template <class T, std::size_t Rank, std::size_t N>
  struct Tensor : Bindable<Tensor<T, Rank, N>>
  {
    using tensor_tag = void;
    using Bindable<Tensor<T, Rank, N>>::operator();

    constexpr static RowMajor<Rank, N> _map;

    T _data[pow(N, Rank)];


    constexpr static auto rank() -> std::size_t
    {
      return Rank;
    }

    constexpr static auto dim() -> std::size_t
    {
      return N;
    }

    constexpr Tensor() = default;

    constexpr Tensor(std::convertible_to<T> auto t, std::convertible_to<T> auto... ts)
      requires((sizeof...(ts) < pow(N, Rank)))
        : _data { T(t), T(ts)... }
    {
    }

    template <class B>
    requires(is_tree<B> and not is_tensor<B> and rank_v<B> == Rank)
    constexpr Tensor(B&& b)
    {
      albert::evaluate(Bind<Tensor, outer_v<B>>(*this), FWD(b));
    }

    template <is_tree B>
    requires(rank_v<B> == Rank)
    constexpr auto operator=(B&& b) & -> Tensor&
    {
      albert::evaluate(Bind<Tensor, outer_v<B>>(*this), FWD(b));
      return *this;
    }

    template <is_tree B>
    requires(rank_v<B> == Rank)
    constexpr auto operator=(B&& b) && -> Tensor&&
    {
      albert::evaluate(Bind<Tensor, outer_v<B>>(std::move(*this)), FWD(b));
      return std::move(*this);
    }

    constexpr auto operator()(std::integral auto i, std::integral auto... is) const
      -> decltype(auto)
      requires(sizeof...(is) + 1 == Rank)
    {
      return _data[_map(i, is...)];
    }

    constexpr auto operator()(std::integral auto i, std::integral auto... is)
      -> decltype(auto)
      requires(sizeof...(is) + 1 == Rank)
    {
      return _data[_map(i, is...)];
    }

    constexpr auto evaluate(ScalarIndex<Rank> const& index) const
      -> decltype(auto)
    {
      puts("hello");
      return _data[_map(index)];
    }

    constexpr auto evaluate(ScalarIndex<Rank> const& index)
      -> decltype(auto)
    {
      puts("world");
      return _data[_map(index)];
    }
  };
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
