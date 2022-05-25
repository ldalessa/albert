#pragma once

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/TensorLayout.hpp"
#include "albert/TensorStorage.hpp"
#include "albert/concepts.hpp"
#include "albert/utils/FWD.hpp"
#include "albert/utils/max.hpp"

namespace albert
{
    template <is_expression A, is_expression B>
    [[gnu::noinline]]
    constexpr auto evaluate(A&& a, B&& b, auto&& op) -> decltype(auto)
    {
        static_assert(is_permutation(outer_v<A>, outer_v<B>));
        static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);

        constexpr TensorIndex l = outer_v<A>;
        constexpr TensorIndex r = outer_v<B>;
        constexpr int Order = order_v<A>;
        constexpr int N = utils::max(dim_v<A>, dim_v<B>);

        ScalarIndex<Order> i;
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

    template <is_expression A, is_expression B>
    [[gnu::noinline]]
    constexpr auto evaluate_via_temp(A&& a, B&& b, auto&& op) -> decltype(auto)
    {
        static_assert(is_permutation(outer_v<A>, outer_v<B>));
        static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);

        constexpr TensorIndex l = outer_v<A>;
        constexpr TensorIndex r = outer_v<B>;
        constexpr int Order = order_v<A>;
        constexpr int N = utils::max(dim_v<A>, dim_v<B>);
        using T = scalar_type_t<A>;

        constexpr RowMajor<Order, N> map = {};
        DenseStorage<T, Order, N> temp;

        { // first evaluate into the temp storage
            ScalarIndex<Order> i;
            do {
                if constexpr (l == r) {
                    temp[map(i)] = b.evaluate(i);
                }
                else {
                    temp[map(i)] = b.evaluate(select<l, r>(i));
                }
            } while (carry_sum_inc<N>(i));
        }

        { // copy out (or accumulate) to the left-hand-side
            ScalarIndex<Order> i;
            do {
                op(a.evaluate(i), temp[map(i)]);
            } while (carry_sum_inc<N>(i));
        }

        return FWD(a);
    }
}
