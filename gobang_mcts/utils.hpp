#pragma once

#include <iostream>
#include <string>

#ifdef NDEBUG
#define assertMsg(condition, msg) ;
#else
#define assertMsg(condition, msg) assertMsgImpl(condition, msg, __FILE__, __LINE__)
#endif

void assertMsgImpl(bool condition, const std::string &msg,
                   const std::string &file, int line)
{
    if (!condition)
    {
        std::cerr << "Assertion failed: " << msg << std::endl;
        std::cerr << "File: " << file << ", line: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}