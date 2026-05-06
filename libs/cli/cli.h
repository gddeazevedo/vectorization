#pragma once

#include <string>

class CLI {
public:
    CLI(int argc, char **argv);
    int run();

private:
    int argc;
    char **argv;
    std::string operation;
    std::string compiler;

    void print_usage(const char *prog);
    int run_spmv();
    int run_ilu();
};