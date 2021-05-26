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
  template <class T, int Rank, int N>
  struct Tensor : Bindable<Tensor<T, Rank, N>>
  {
    using tensor_tag = void;
    using Bindable<Tensor<T, Rank, N>>::operator();

    constexpr static RowMajor<Rank, N> _map = {};

    T _data[pow(N, Rank)];

    constexpr static auto rank() -> int
    {
      return Rank;
    }

    constexpr static auto dim() -> int
    {
      return N;
    }

    constexpr Tensor() = default;

    constexpr Tensor(std::convertible_to<T> auto t, std::convertible_to<T> auto... ts)
      requires((sizeof...(ts) < pow(N, Rank)))
        : _data { T(t), T(ts)... }
    {
    }

    template <is_tree B>
    requires(not is_tensor<B> and not std::convertible_to<B, T> and rank_v<B> == Rank)
    constexpr Tensor(B&& b)
    {
      static_assert(outer_v<B>.repeated().size() == 0);
      static_assert(outer_v<B>.scalars().size() == 0);
      albert::evaluate(Bind(*this, {}, nttp<outer_v<B>>), FWD(b));
    }

    template <is_tree B>
    requires(rank_v<B> == Rank)
    constexpr auto operator=(B&& b) & -> decltype(auto)
    {
      return std::move(Bind(*this, {}, nttp<outer_v<B>>) = FWD(b));
    }

    template <is_tree B>
    requires(rank_v<B> == Rank)
    constexpr auto operator=(B&& b) && -> decltype(auto)
    {
      return std::move(Bind(std::move(*this), {}, nttp<outer_v<B>>) = FWD(b));
    }

    template <class... Is>
    requires(all_integral_index<Is...> and sizeof...(Is) + 1 == Rank)
    constexpr auto operator()(std::integral auto i, Is... is) const &
      -> decltype(auto)
    {
      return _data[_map(i, is...)];
    }

    template <class... Is>
    requires(all_integral_index<Is...> and sizeof...(Is) + 1 == Rank)
    constexpr auto operator()(std::integral auto i, Is... is) &
      -> decltype(auto)
    {
      return _data[_map(i, is...)];
    }

    constexpr auto evaluate(ScalarIndex<Rank> const& index) const
      -> decltype(auto)
    {
      return _data[_map(index)];
    }

    constexpr auto evaluate(ScalarIndex<Rank> const& index)
      -> decltype(auto)
    {
      return _data[_map(index)];
    }
  };
}

#endif // ALBERT_INCLUDE_TENSOR_HPP
