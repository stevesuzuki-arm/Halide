#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace Halide::Internal;

void example() {
    Var x("x");
    Expr expr = 4 + min(x, x) + 12;
    std::cout << "I expect this to simplify to `x + 16`!\n";
    Expr simpl = simplify(expr);
    std::cout << "Original: " << expr << "\n";
    std::cout << "Simplified: " << simpl << "\n";
}

int main(int argc, char **argv) {
    example();
    return 0;
}