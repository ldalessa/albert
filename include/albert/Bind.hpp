#ifndef ALBERT_INCLUDE_BIND_HPP
#define ALBERT_INCLUDE_BIND_HPP

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/TensorIndex.hpp"
#include "albert/concepts.hpp"
#include "albert/evaluate.hpp"
#include "albert/traits.hpp"
#include "albert/utils.hpp"
#include <ce/cvector.hpp>
#include <utility>

namespace albert
{
  template <class T>
  struct Bindable;

  /// The bind node.
  ///
  /// A bind node associates some subtree with an Einstein notation index
  /// mapping, potentially including self-contraction, and/or a set of
  /// projected indices.
  ///
  /// Bind nodes may have full subtrees or just leafs (like tensor nodes).
  ///
  /// Bind nodes look like:
  ///
  ///     is_index i, j, k, l;                     // indices
  ///     is_tensor A;                             // actual matrix tensor
  ///     is_tree B;                               // matrix tree
  ///     is_tree C;                               // vector tree
  ///     auto bind = A(i, j);                     // basic bind
  ///     auto bind = A(i, i);                     // trace
  ///     auto bind = A(1, i);                     // projection
  ///     auto bind = A(1, 0);                     // fully projected
  ///
  ///     auto product = A(i, j) * B(j);           // normal vector product
  ///     auto product = A(i, k) * B(k, j);        // normal matrix product
  ///     auto product = A(i, j) * B(i, j);        // inner product
  ///     auto product = A(i, j) * B(k, l);        // kronecker product
  ///     auto product = A(k, i) * B(j, k);        // some odd contraction
  ///     auto product = A(i, j) * B(j, k) * C(k); // expression
  ///     auto product = (A(i, j) + B(j, i)) / 2;  // symmetric transformation
  ///
  ///     A(i, j) = B(j, i);                       // transpose assignment
  ///     A(1, j) = B(j, 0);                       // projection assignment
  ///     A(0, i) = B(i, j) * C(j);                // projection assignment
  ///     A(0, 0) = 42;                            // scalar assignment
  ///
  /// @param     A The type of the subtree.
  /// @param index The index binding to the subtree.
  template <is_tree A, is_tensor_index auto index>
  struct Bind : Bindable<Bind<A, index>>
  {
    /// Subscribe to tree concepts.
    /// @{
    using tree_node_tag = void;
    using unary_node_tag = void;
    /// @}

    constexpr static int M = index.scalars().size(); //!< number of projected indices

    A a;                                        //!< subtree
    ScalarIndex<M> _scalars;                    //!< projected indices

    /// Construct a bind node for a subtree.
    ///
    /// @param       a The subtree we're binding.
    /// @param scalars Scalars associated with projections (if any).
    /// @param       ? Helper to specify the `index` CNTTP.
    constexpr Bind(A a, ce::cvector<int, M> const& scalars, nttp_args<index>)
        : a(std::move(a))
        , _scalars(scalars)
    {
      constexpr auto l = rank_v<A>;
      constexpr auto r = index.size();
      static_assert(l == r);
    }

    /// Default copy and move will prevent implicit operator= generation, which
    /// means that the `is_tree` version will match _all_ instances of
    /// Bind::operator=.
    /// @{
    constexpr Bind(Bind const&) = default;
    constexpr Bind(Bind&&) = default;
    /// @}

    template <is_tree B>
    constexpr Bind& operator=(B&& b)
    {
      constexpr TensorIndex l = outer_v<Bind>;
      constexpr TensorIndex r = outer_v<B>;
      static_assert(is_permutation(l, r));
      return albert::evaluate(*this, FWD(b), [](auto&& a, auto&&b) {
        FWD(a) = FWD(b);
      });
    }

    ///
    constexpr static auto outer()
    {
      return index.exclusive();
    }

    constexpr static auto dim() -> decltype(auto)
    {
      return dim_v<A>;
    }

    /// Evaluate a bind node when there's a contraction.
    ///
    /// Contracted binds can only exist on the right-hand-side of an equation,
    /// because they don't represent lvalues when evaluated.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> const& i) const
      -> auto requires(index.repeated().size() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.scalars();
      constexpr TensorIndex  inner = index.repeated();
      constexpr TensorIndex    all = outer + slices + inner;
      constexpr int      N = dim();
      constexpr int   Rank = outer.size() + slices.size();
      constexpr int      I = inner.size();

      auto rhs = [&](auto const& i) {
        return a.evaluate(select<all, index>(i));
      };

      ScalarIndex<Rank + I> j(i + _scalars);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Rank>(j));
      return temp;
    }

    /// Evaluate a bind node when there's only a projection.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> const& i) const
      -> decltype(auto) requires(index.repeated().size() == 0 and
                                 index.scalars().size() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.scalars();
      constexpr TensorIndex    all = outer + slices;
      return a.evaluate(select<all, index>(i + _scalars));
    }

    /// Evaluate a bind node when there's only a projection.
    ///
    /// This version of the bind represents an assignable lvalue, assuming that
    /// the underlying expression is assignable. This is currently only true for
    /// raw tensors.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> const& i)
      -> decltype(auto) requires(index.repeated().size() == 0 and
                                 index.scalars().size() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.scalars();
      constexpr TensorIndex    all = outer + slices;
      return a.evaluate(select<all, index>(i + _scalars));
    }

    /// Evaluate a bind that contains neither contraction nor projection.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) const -> decltype(auto)
      requires((index.scalars().size() + index.repeated().size()) == 0)
    {
      return a.evaluate(i);
    }

    /// Evaluate a bind that contains neither contraction nor projection.
    ///
    /// This version of the bind represents an assignable lvalue, assuming that
    /// the underlying expression is assignable. This is currently only true for
    /// raw tensors.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) -> decltype(auto)
      requires((index.scalars().size() + index.repeated().size()) == 0)
    {
      return a.evaluate(i);
    }
  };

  template <class T>
  struct Bindable
  {
    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(I i, Is... is) const & -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        *static_cast<T const*>(this),
        scalars,
        nttp<index>
      };
    }

    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(I i, Is... is) && -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        std::move(*static_cast<T*>(this)),
        scalars,
        nttp<index>
      };
    }

    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(is_index auto i, Is... is) & -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        *static_cast<T*>(this),
        scalars,
        nttp<index>
      };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() const & -> decltype(auto)
    {
      return Bind { *static_cast<T const*>(this), {}, nttp<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() && -> decltype(auto)
    {
      return Bind { std::move(*static_cast<T*>(this)), {}, nttp<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() & -> decltype(auto)
    {
      return Bind{ *static_cast<T*>(this), {}, nttp<index> };
    }
  };
}

#endif // ALBERT_INCLUDE_BIND_HPP
