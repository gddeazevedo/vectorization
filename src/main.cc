#include <iostream>
#include "bcsr.h"


int main(int argc, char **argv) {
    BlockedCSR *A = generate_blocked27_3x3(3, 3, 3);
    bc_draw(A);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
