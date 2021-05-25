#ifndef ALBERT_INCLUDE_CONCEPTS_HPP
#define ALBERT_INCLUDE_CONCEPTS_HPP

#include <concepts>

namespace albert
{
  template <class T>
  concept is_tree_node = requires {
    typename std::remove_cvref_t<T>::tree_node_tag;
  };

  template <class T>
  concept is_binary_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::binary_node_tag;
  };

  template <class T>
  concept is_unary_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::unary_node_tag;
  };

  template <class T>
  concept is_leaf_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::leaf_node_tag;
  };

  template <class T>
  concept is_tensor = requires {
    typename std::remove_cvref_t<T>::tensor_tag;
  };

  template <class T>
  concept is_tree = (is_tree_node<T> ||
                     is_tensor<T> ||
                     std::integral<T> ||
                     std::floating_point<T>);

  template <class T>
  concept is_index = std::integral<T> or requires {
    typename std::remove_cvref_t<T>::index_tag;
  };

  template<typename... Ts>
  concept all_index = (is_index<Ts> && ...);

  template<typename... Ts>
  concept all_integral_index = all_index<Ts...> && (std::integral<Ts> && ...);

  template <class T>
  concept is_tensor_index = requires {
    typename std::remove_cvref_t<T>::tensor_index_tag;
  };
}

#endif // ALBERT_INCLUDE_CONCEPTS_HPP
