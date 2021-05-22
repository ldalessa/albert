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
  auto e = pow(sin(tanh(exp(D(symmetrize(D((a(i) + b(i))(j),i) * c), i)))), d);

  // print<decltype(d)> _;
}
