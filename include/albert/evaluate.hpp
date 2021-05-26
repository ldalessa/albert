#ifndef ALBERT_INCLUDE_EVALUATE_HPP
#define ALBERT_INCLUDE_EVALUATE_HPP

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/concepts.hpp"
#include "albert/traits.hpp"
#include "albert/utils.hpp"

namespace albert
{
  template <is_tree A, is_tree B, class Op>
  [[gnu::noinline]]
  constexpr auto evaluate(A&& a, B&& b, Op&& op) -> decltype(auto)
  {
    static_assert(is_permutation(outer_v<A>, outer_v<B>));
    static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);

    constexpr TensorIndex l = outer_v<A>;
    constexpr TensorIndex r = outer_v<B>;
    constexpr int Rank = rank_v<A>;
    constexpr int N = max(dim_v<A>, dim_v<B>);

    ScalarIndex<Rank> i;
    do {
      if constexpr (l == r) {
        op(a.evaluate(i), b.evaluate(i));
      }
      else {
        op(a.evaluate(i), b.evaluate(select<l, r>(i)));
      }
    } while (carry_sum_inc<N>(i));

    return FWD(a);
  }
}

#endif // ALBERT_INCLUDE_EVALUATE_HPP
