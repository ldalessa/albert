#ifndef ALBERT_INCLUDE_EXPRESSIONS_HPP
#define ALBERT_INCLUDE_EXPRESSIONS_HPP

#include "albert/Bind.hpp"
#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/cmath.hpp"
#include "albert/concepts.hpp"
#include "albert/utils.hpp"
#include <utility>

namespace albert
{
  template <is_tree A, is_tree B>
  struct Addition
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

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

    constexpr static auto outer()
    {
      return outer_v<A>;
    }

    constexpr static auto dim() -> std::size_t
    {
      return max(dim_v<A>, dim_v<B>);
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Addition>> const& i, auto&& op) const
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

  template <is_tree A, is_tree B>
  struct Sum : Addition<A, B>, Bindable<Sum<A, B>>
  {
    using Addition<A, B>::Addition;

    constexpr auto evaluate(ScalarIndex<rank_v<Sum>> const& i) const
    {
      return Addition<A, B>::evaluate(i, [](auto&& a, auto&& b) {
        return FWD(a) + FWD(b);
      });
    }
  };

  template <is_tree A, is_tree B>
  Sum(A, B) -> Sum<A, B>;

  template <is_tree A, is_tree B>
  struct Diff : Addition<A, B>, Bindable<Diff<A, B>>
  {
    using Addition<A, B>::Addition;

    constexpr auto evaluate(ScalarIndex<rank_v<Diff>> const& i) const
    {
      return Addition<A, B>::evaluate(i, [](auto&& a, auto&& b) {
        return FWD(a) - FWD(b);
      });
    }
  };

  template <is_tree A, is_tree B>
  Diff(A, B) -> Diff<A, B>;

  template <is_tree A, is_tree B>
  struct Contraction
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

    A a;
    B b;

    constexpr Contraction(A a, B b)
        : a(std::move(a))
        , b(std::move(b))
    {
      static_assert(dim_v<A> == 0 || dim_v<B> == 0 || dim_v<A> == dim_v<B>);
    }

    constexpr static auto outer()
    {
      constexpr TensorIndex a = outer_v<A>;
      constexpr TensorIndex b = outer_v<B>;
      constexpr TensorIndex c = a ^ b;
      return c;
    }

    constexpr static auto dim() -> std::size_t
    {
      return max(dim_v<A>, dim_v<B>);
    }
  };

  template <is_tree A, is_tree B>
  struct Product : Contraction<A, B>, Bindable<Product<A, B>>
  {
    using Contraction<A, B>::Contraction;
    using Contraction<A, B>::a;
    using Contraction<A, B>::b;
    using Contraction<A, B>::dim;

    constexpr auto evaluate(ScalarIndex<rank_v<Product>> const& i) const
      -> auto
    {
      constexpr TensorIndex outer = outer_v<Product>;
      constexpr TensorIndex     l = outer_v<A>;
      constexpr TensorIndex     r = outer_v<B>;
      constexpr TensorIndex inner = l & r;
      constexpr TensorIndex   all = outer + inner;
      constexpr std::size_t     N = dim();
      constexpr std::size_t  Rank = outer.size();
      constexpr std::size_t     I = inner.size();

      auto rhs = [&](auto const& index) {
        return a.evaluate(select<all, l>(index)) * b.evaluate(select<all, r>(index));
      };

      ScalarIndex<Rank + I> j(i);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Rank>(j));
      return temp;
    }
  };

  template <is_tree A, is_tree B>
  Product(A, B) -> Product<A, B>;

  template <is_tree A, is_tree B>
  struct Ratio : Contraction<A, B>
  {
    using Contraction<A, B>::Contraction;
    using Contraction<A, B>::a;
    using Contraction<A, B>::b;
    using Contraction<A, B>::dim;

    constexpr auto evaluate(ScalarIndex<rank_v<Ratio>> const& i) const
    {
      __builtin_abort();
      constexpr TensorIndex outer = outer_v<Ratio>;
      constexpr TensorIndex     l = outer_v<A>;
      constexpr TensorIndex     r = outer_v<B>;
      constexpr TensorIndex inner = l & r;
      constexpr TensorIndex   all = outer + inner;
      constexpr std::size_t     N = dim();
      constexpr std::size_t  Rank = outer.size();
      constexpr std::size_t     I = inner.size();

      auto rhs = [&](auto const& index) {
        return a.evaluate(select<all, l>(index)) * b.evaluate(select<all, r>(index));
      };

      ScalarIndex<Rank + I> j(i);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Rank>(j));
      return temp;
    }
  };

  template <is_tree A, is_tree B>
  Ratio(A, B) -> Ratio<A, B>;

  template <is_tree A>
  struct Negate : Bindable<Negate<A>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr Negate(A a)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }

    constexpr static auto dim() -> std::size_t
    {
      return dim_v<A>;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Negate>> const& i) const
    {
      return -a.evaluate(i);
    }
  };

  template <is_tree A, is_tensor_index auto index>
  struct Partial : Bindable<Partial<A, index>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr Partial(A a)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return (outer_v<A> + index).exclusive();
    }

    constexpr static auto dim() -> std::size_t
    {
      return dim_v<A>;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Partial>> const&) const;
  };

  template <class T>
  struct Literal
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    T x;

    constexpr Literal(T x)
        : x(x)
    {
    }

    constexpr static auto rank() -> std::size_t
    {
      return 0;
    }

    constexpr static auto dim() -> std::size_t
    {
      return 0;
    }

    constexpr static auto outer() -> TensorIndex<0>
    {
      return {};
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Literal>> const&) const
      -> T const&
    {
      return x;
    }
  };

  template <TensorIndex<2> index>
  struct Delta : Bindable<Delta<index>>
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    constexpr static auto rank() -> std::size_t
    {
      return 2;
    }

    constexpr static auto dim() -> std::size_t
    {
      return 0;
    }

    constexpr static auto outer() -> TensorIndex<2>
    {
      return index;
    }

    constexpr static auto evaluate(ScalarIndex<rank_v<Delta>> const& i)
      -> bool
    {
      return i[0] == i[1];
    }
  };

  template <is_tensor_index auto index>
  struct Epsilon : Bindable<Delta<index>>
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    constexpr static auto rank() -> std::size_t
    {
      return index.size();
    }

    constexpr static auto dim() -> std::size_t
    {
      return 0;
    }

    constexpr static auto outer()
    {
      return index;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Epsilon>> const&) const;
  };
}

#endif // ALBERT_INCLUDE_EXPRESSIONS_HPP
