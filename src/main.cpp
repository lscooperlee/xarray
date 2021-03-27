#include "cmdparser.h"
#include <iostream>

int main(int argc, const char* const argv[])
{
    auto Commands = std::make_tuple(
        Command<>({ "-h", "--help" }, "help", []() { std::cout << "print help" << std::endl; }));

    ParseCommandOpt(argc, argv, Commands);

    RunCommand(Commands);

    return 0;
}
