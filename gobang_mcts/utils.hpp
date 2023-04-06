#pragma once

#include <iostream>
#include <string>

void assertMsg(bool condition, const std::string &msg,
               const std::string &file, int line)
{
    if (!condition)
    {
        std::cerr << "Assertion failed: " << msg << std::endl;
        std::cerr << "File: " << file << ", line: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}