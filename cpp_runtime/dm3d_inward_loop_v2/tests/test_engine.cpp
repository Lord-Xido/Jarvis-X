#include "dm3d/core.hpp"
#include <iostream>
#include <exception>
int main() { try { dm3d::run_selftest(); return 0; }
catch(const std::exception& e) { std::cerr << "TEST FAILURE: " << e.what() << "\n"; return 1; } }
