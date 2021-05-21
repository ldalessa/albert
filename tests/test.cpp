#include "albert/albert.hpp"

template <class> struct print;

using namespace albert;
int main() {
  Index<'i'> i;
  Index<'j'> j;
  Tensor<double, 1, 3> a;
  Tensor<double, 1, 3> b;
  Tensor<double, 0, 3> c;
  auto d = ((a(i) + b(i)) * δ(i,j)) * b(j) + 1;
  auto e = 3.1415 + D((a(i) + b(i))(j),j) * c + 1;

  // print<decltype(d)> _;
}
