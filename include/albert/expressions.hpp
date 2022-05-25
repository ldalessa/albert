#pragma once

#include "albert/Bind.hpp"
#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/cmath.hpp"
#include "albert/solver.hpp"
#include "albert/concepts.hpp"
#include "albert/concepts/tensor_index.hpp"
#include "albert/traits/scalar_type.hpp"
#include "albert/utils/FWD.hpp"
#include "albert/utils/max.hpp"
#include <bit>
#include <utility>

namespace albert
{
    template <is_expression A, is_expression B>
    struct Addition
    {
        A a;
        B b;

        constexpr Addition(A a, B b)
                : a(std::move(a))
                , b(std::move(b))
        {
            constexpr auto l_index = outer_v<A>;
            constexpr auto r_index = outer_v<B>;
            static_assert(is_permutation(l_index, r_index)); // tensor expression addition must have compatible indices
            static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);
        }

        constexpr static bool contains(auto&& tag)
        {
            return A::contains(FWD(tag)) || B::contains(FWD(tag));
        }

        constexpr static bool may_alias(auto&& tag)
        {
            return (A::may_alias(FWD(tag)) ||
                    B::may_alias(FWD(tag)) ||
                    (outer_v<A> != outer_v<B> and contains(FWD(tag))));
        }

        constexpr static auto order() -> int
        {
            return order_v<A>;
        }

        constexpr static auto dim() -> int
        {
            return utils::max(dim_v<A>, dim_v<B>);
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return outer_v<A>;
        }

        constexpr auto evaluate(ScalarIndex<order_v<Addition>> const& i, auto&& op) const
        {
            constexpr TensorIndex l = outer_v<A>;
            constexpr TensorIndex r = outer_v<B>;
            if constexpr (l == r) {
                return FWD(op)(a.evaluate(i), b.evaluate(i));
            }
            else {
                return FWD(op)(a.evaluate(i), b.evaluate(select<l, r>(i)));
            }
        }
    };

    template <is_expression A, is_expression B>
    struct Sum : Addition<A, B>, Bindable<Sum<A, B>>
    {
        using Addition<A, B>::Addition;

        using scalar_type = decltype(std::declval<traits::scalar_type_t<A>>() + std::declval<traits::scalar_type_t<B>>());

        /// Evaluate into a scalar.
        constexpr operator scalar_type() const requires (order_v<Sum> == 0)
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr auto evaluate(ScalarIndex<order_v<Sum>> const& i) const
        {
            return Addition<A, B>::evaluate(i, [](auto&& a, auto&& b) {
                return FWD(a) + FWD(b);
            });
        }
    };

    template <is_expression A, is_expression B>
    Sum(A, B) -> Sum<A, B>;

    template <is_expression A, is_expression B>
    struct Diff : Addition<A, B>, Bindable<Diff<A, B>>
    {
        using Addition<A, B>::Addition;

        using scalar_type = decltype(std::declval<traits::scalar_type_t<A>>() - std::declval<traits::scalar_type_t<B>>());

        /// Evaluate into a scalar.
        constexpr operator scalar_type() const requires (order_v<Diff> == 0)
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr auto evaluate(ScalarIndex<order_v<Diff>> const& i) const
        {
            return Addition<A, B>::evaluate(i, [](auto&& a, auto&& b) {
                return FWD(a) - FWD(b);
            });
        }
    };

    template <is_expression A, is_expression B>
    Diff(A, B) -> Diff<A, B>;

    template <is_expression A, is_expression B>
    struct Product : Bindable<Product<A, B>>
    {
        using scalar_type = decltype(std::declval<traits::scalar_type_t<A>>() * std::declval<traits::scalar_type_t<B>>());

        A a;
        B b;

        constexpr Product(A a, B b)
                : a(std::move(a))
                , b(std::move(b))
        {
            static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);
        }

        constexpr static bool contains(auto&& tag)
        {
            return A::contains(FWD(tag)) || B::contains(FWD(tag));
        }

        constexpr static bool may_alias(auto&& tag)
        {
            // if there's a contraction and one of my children contains the tag then
            // we may alias
            return ((outer_v<A> & outer_v<B>).size() and contains(FWD(tag)));
        }

        /// Evaluate into a scalar.
        constexpr operator scalar_type() const requires (order_v<Product> == 0)
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr static auto order() -> int
        {
            return outer().size();
        }

        constexpr static auto dim() -> int
        {
            return utils::max(dim_v<A>, dim_v<B>);
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            constexpr TensorIndex a = outer_v<A>;
            constexpr TensorIndex b = outer_v<B>;
            constexpr TensorIndex c = a ^ b;
            return c;
        }

        constexpr auto evaluate(ScalarIndex<order_v<Product>> const& i) const
            -> auto
        {
            constexpr TensorIndex outer = outer_v<Product>;
            constexpr TensorIndex     l = outer_v<A>;
            constexpr TensorIndex     r = outer_v<B>;
            constexpr TensorIndex inner = l & r;
            constexpr TensorIndex   all = outer + inner;
            constexpr int     N = dim();
            constexpr int Order = outer.size();
            constexpr int     I = inner.size();

            auto rhs = [&](auto const& index) {
                return a.evaluate(select<all, l>(index)) * b.evaluate(select<all, r>(index));
            };

            ScalarIndex<Order + I> j(i);
            decltype(rhs(j)) temp{};
            do {
                temp += rhs(j);
            } while (carry_sum_inc<N, Order>(j));
            return temp;
        }
    };

    /// Represent a scalar division operation for an integral divisor.
    ///
    /// For floating point and higher order tensor division the grammar has
    /// transformed the operator/ to an operator* and an inverse expression, but
    /// we can't eagerly invert an integral properly. For example
    ///
    ///   Tensor A = { 2 };
    ///   int c = 2;
    ///   Tensor B = A() / 2;
    ///            // -> A() * inverse(2);
    ///            // -> A() * (1 / 2);
    ///            // -> A() * 0;
    ///            // -> 0;
    template <is_expression A, std::integral B>
    struct Ratio : Bindable<Ratio<A, B>>
    {
        using scalar_type = decltype(std::declval<traits::scalar_type_t<A>>() * std::declval<traits::scalar_type_t<B>>());

        A a;
        B b;

        constexpr Ratio(A a, B b)
                : a(std::move(a))
                , b(b)
        {
        }

        constexpr static bool contains(auto&& tag)
        {
            return A::contains(FWD(tag));
        }

        constexpr static bool may_alias(auto&& tag)
        {
            return A::may_alias(FWD(tag));
        }

        /// Evaluate into a scalar.
        constexpr operator scalar_type() const requires (order_v<A> == 0)
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr static auto order() -> int
        {
            return order_v<A>;
        }

        constexpr static auto dim() -> int
        {
            return dim_v<A>;
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return outer_v<A>;
        }

        constexpr auto evaluate(ScalarIndex<Ratio::order()> const& i) const
            -> auto
        {
            return a.evaluate(i) / b;
        }
    };

    template <is_expression A>
    struct Negate : Bindable<Negate<A>>
    {
        using scalar_type = traits::scalar_type_t<A>;

        A a;

        constexpr Negate(A a)
                : a(std::move(a))
        {
        }

        constexpr static bool contains(auto&& tag)
        {
            return A::contains(FWD(tag));
        }

        constexpr static bool may_alias(auto&& tag)
        {
            return A::may_alias(FWD(tag));
        }

        constexpr static auto order() -> int
        {
            return order_v<A>;
        }

        constexpr static auto dim() -> int
        {
            return dim_v<A>;
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return outer_v<A>;
        }

        constexpr auto evaluate(ScalarIndex<order_v<Negate>> const& i) const
        {
            return -a.evaluate(i);
        }
    };

    template <is_expression A, concepts::tensor_index auto index>
    struct Partial : Bindable<Partial<A, index>>
    {
        using scalar_type = traits::scalar_type_t<A>;

        A a;

        constexpr Partial(A a)
                : a(std::move(a))
        {
        }

        constexpr static bool contains(auto&& tag)
        {
            assert(not A::contains(FWD(tag)));
            return false;
        }

        constexpr static bool may_alias(auto&& tag)
        {
            assert(not A::contains(FWD(tag)));
            return false;
        }

        /// Evaluate into a scalar.
        constexpr operator scalar_type() const requires (order_v<Partial> == 0)
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr static auto order() -> int
        {
            return outer().size();
        }

        constexpr static auto dim() -> int
        {
            return dim_v<A>;
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return (outer_v<A> + index).exclusive();
        }

        constexpr auto evaluate(ScalarIndex<order_v<Partial>> const&) const;
    };

    template <is_expression A>
    struct Inverse : Bindable<Inverse<A>>
    {
        using scalar_type = traits::scalar_type_t<A>;

        A a;

        constexpr Inverse(A a)
                : a(std::move(a))
        {
        }

        constexpr static bool contains(auto&& tag)
        {
            return A::contains(FWD(tag));
        }

        constexpr static bool may_alias(auto&& tag)
        {
            return A::may_alias(FWD(tag));
        }

        constexpr static auto order() -> int
        {
            return order_v<A>;
        }

        constexpr static auto dim() -> int
        {
            return dim_v<A>;
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return outer_v<A>;
        }

        constexpr auto evaluate(ScalarIndex<order_v<Inverse>> const&) const;
    };

    template <is_expression A> requires(order_v<A> == 0)
        struct Inverse<A> : Bindable<Inverse<A>>
        {
            using scalar_type = traits::scalar_type_t<A>;

            A a;

            constexpr Inverse(A a)
                    : a(std::move(a))
            {
            }

            constexpr static bool contains(auto)
            {
                return false;
            }

            constexpr static bool may_alias(auto)
            {
                return false;
            }

            /// Evaluate into a scalar.
            constexpr operator scalar_type()
            {
                return evaluate(ScalarIndex<0>{});
            }

            constexpr static auto order() -> int
            {
                return order_v<A>;
            }

            constexpr static auto dim() -> int
            {
                return dim_v<A>;
            }

            constexpr static auto outer() -> concepts::tensor_index auto
            {
                return outer_v<A>;
            }

            constexpr auto evaluate(ScalarIndex<0> const& i) const
            {
                using T = decltype(a.evaluate(i));
                return T(1) / a.evaluate(i);
            }
        };

    template <class T>
    struct Literal
    {
        using scalar_type = traits::scalar_type_t<T>;

        T x;

        constexpr Literal(T x)
                : x(x)
        {
        }

        constexpr static bool contains(auto)
        {
            return false;
        }

        constexpr static bool may_alias(auto)
        {
            return false;
        }

        /// Evaluate into a scalar.
        constexpr operator scalar_type()
        {
            return evaluate(ScalarIndex<0>{});
        }

        constexpr static auto order() -> int
        {
            return 0;
        }

        constexpr static auto dim() -> int
        {
            return 0;
        }

        constexpr static auto outer() -> TensorIndex<0>
        {
            return {};
        }

        constexpr auto evaluate(ScalarIndex<order_v<Literal>> const&) const
            -> T const&
        {
            return x;
        }
    };

    template <TensorIndex<2> index>
    struct Delta : Bindable<Delta<index>>
    {
        using scalar_type = int;

        constexpr static bool contains(auto)
        {
            return false;
        }

        constexpr static bool may_alias(auto)
        {
            return false;
        }

        constexpr static auto order() -> int
        {
            return 2;
        }

        constexpr static auto dim() -> int
        {
            return 0;
        }

        constexpr static auto outer() -> TensorIndex<2>
        {
            return index;
        }

        constexpr static auto evaluate(ScalarIndex<order_v<Delta>> const& i)
            -> int
        {
            return i[0] == i[1];
        }
    };

    template <concepts::tensor_index auto index>
    struct LeviCivita : Bindable<LeviCivita<index>>
    {
        using scalar_type = int;

        constexpr static bool contains(auto)
        {
            return false;
        }

        constexpr static bool may_alias(auto)
        {
            return false;
        }

        constexpr static auto order() -> int
        {
            return index.size();
        }

        constexpr static auto dim() -> int
        {
            return order();
        }

        constexpr static auto outer() -> concepts::tensor_index auto
        {
            return index;
        }

        constexpr auto evaluate(ScalarIndex<order_v<LeviCivita>> const& i) const
            -> int
        {
            static_assert(0 <= order() and order() < 64);
            uint64_t n = 0;
            for (auto i : i) {
                n |= 1u << i;
            }

            return (std::popcount(n) != order()) ? 0 : parity(i);
        }

        constexpr auto parity(ScalarIndex<order_v<LeviCivita>> perm) const
            -> int
        {
            int swaps = 0;
            for (int i = 0, e = std::size(perm); i < e; ++i) {
                for (int j = std::exchange(perm[i], i); i != j; std::swap(j, perm[j])) {
                    ++swaps;
                }
            }
            return (swaps & 1) ? -1 : 1;
        }

        // Adapted from https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
        // and https://en.cppreference.com/w/cpp/algorithm/next_permutation for random
        // access iterators and parity tracking.
        constexpr auto next_permutation(int parity, auto first, auto last)
            -> int // -1: odd, 1: even, 0: wrapped
        {
            for (auto i = last - 1; i != first;)
            {
                // find the last i s.t. i < i + 1
                if (auto i1 = i; *--i < *i1)
                {
                    // find the last j s.t. i < j
                    auto i2 = last;
                    while (*--i2 < *i);
                    std::iter_swap(i, i2);
                    std::reverse(i1, last);

                    // just swapped 1 + floor((last - first) / 2) elements, invert parity
                    // if that is an odd number
                    int swap = (last - i1 + 2) & 2; // 0 or 2 (random access)
                    parity += 1;                    // 0 or 2
                    parity ^= swap;                 // 0 or 2
                    return parity - 1;              // 1 or -1
                }
            }

            std::reverse(first, last);
            return 0;
        }
    };
}
