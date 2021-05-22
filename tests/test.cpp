#include "albert/albert.hpp"

template <class> struct print;

using namespace albert::grammar;
int main() {
  albert::Index<'i'> i;
  albert::Index<'j'> j;
  albert::Tensor<double, 1, 3> a;
  albert::Tensor<double, 1, 3> b;
  albert::Tensor<double, 0, 3> c;
  auto d = ((a(i) + b(i)) * δ(i,j)) * b(j) + 1;
  auto e = pow(sin(tanh(exp(D(symmetrize(D((a(i) + b(i))(j),i) * c), i)))), d);

  c = d;

  // print<decltype(d)> _;
}
