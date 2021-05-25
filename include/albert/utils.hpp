#ifndef ALBERT_INCLUDE_UTILS_HPP
#define ALBERT_INCLUDE_UTILS_HPP

#define FWD(x) static_cast<decltype(x)&&>(x)

namespace albert
{
  constexpr int pow(int M, int D)
  {
    int out = 1;
    for (int i = 0; i < D; i++) {
      out *= M;
    }
    return out;
  }

  constexpr int max(int a, int b)
  {
    return (a < b) ? b : a;
  }

  constexpr int min(int a, int b)
  {
    return (a < b) ? a : b;
  }

  template <auto...> struct nttp_args {};
  template <auto... args>
  constexpr inline nttp_args<args...> nttp_pack = {};
}

#endif // ALBERT_INCLUDE_UTILS_HPP
