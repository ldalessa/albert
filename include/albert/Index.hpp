#ifndef ALBERT_INCLUDE_INDEX_HPP
#define ALBERT_INCLUDE_INDEX_HPP

#include <concepts>

namespace albert
{
  template <char... Is>
  requires(sizeof...(Is) > 0)
  struct Index
  {
    using index_tag = void;
  };

  template <class>
  struct to_index_type
  {
    using type = Index<'\0'>;
  };

  template <char I>
  struct to_index_type<Index<I>>
  {
    using type = Index<I>;
  };

  template <class T>
  using to_index_type_t = typename to_index_type<T>::type;

  template <class...>
  struct cat_index_type;

  template <char... Is>
  struct cat_index_type<Index<Is>...>
  {
    using type = Index<Is...>;
  };

  template <class... Ts>
  using cat_index_type_t = typename cat_index_type<to_index_type_t<Ts>...>::type;
}

#endif // ALBERT_INCLUDE_INDEX_HPP
