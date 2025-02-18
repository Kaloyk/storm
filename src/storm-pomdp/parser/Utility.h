#pragma once
#include <string>
#include <vector>
#include <sstream>

// Utility function to split a string based on a delimiter
inline std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    size_t last = str.find_last_not_of(" \t\n\r");
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

template<typename ValueType>
ValueType convertToValueType(const std::string& s) {
    std::istringstream iss(s);
    ValueType val;
    if (!(iss >> val)) {
        throw std::runtime_error("Conversion error: cannot convert \"" + s + "\" to the desired numeric type.");
    }
    return val;
}