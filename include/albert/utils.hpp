#ifndef ALBERT_INCLUDE_UTILS_HPP
#define ALBERT_INCLUDE_UTILS_HPP

#define FWD(x) static_cast<decltype(x)&&>(x)

namespace albert
{
  constexpr std::size_t pow(std::size_t M, std::size_t D)
  {
    std::size_t out = 1;
    for (int i = 0; i < D; i++) {
      out *= M;
    }
    return out;
  }

  constexpr std::size_t max(std::size_t a, std::size_t b)
  {
    return (a < b) ? b : a;
  }

  constexpr std::size_t min(std::size_t a, std::size_t b)
  {
    return (a < b) ? a : b;
  }
}

#endif // ALBERT_INCLUDE_UTILS_HPP
