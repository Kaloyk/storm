#pragma once
#include <sstream>
#include <string>
#include <vector>

// Utility function to split a string based on a delimiter
inline std::vector<std::string> split(const std::string& s, char delimiter) {
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

inline bool isIgnoredLine(const std::string& line) {
    std::string trimmed = trim(line);
    return (trimmed.empty() || trimmed[0] == '#');
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

void parseIdentifierVector(std::istringstream& iss, std::vector<std::string>& vec, const std::string& identifierName) {
    std::string token;
    if (!(iss >> token))
        return;

    // Check if the token is composed only of digits.
    bool is_number = !token.empty() && std::all_of(token.begin(), token.end(), ::isdigit);

    if (is_number) {
        int count = std::stoi(token);
        for (int i = 0; i < count; i++) {
            vec.push_back(std::to_string(i));
        }
    } else {
        // The token is not a number, so treat it as the first identifier.
        vec.push_back(trim(token));
        while (iss >> token) {
            vec.push_back(trim(token));
        }
    }
}

template<typename ValueType>
std::vector<ValueType> readArrayTokens(std::istringstream& iss, std::ifstream& infile) {
    std::vector<ValueType> values;
    std::string token;

    // try reading first line
    while (iss >> token) {
        values.push_back(convertToValueType<ValueType>(token));
    }

    // if empty try reading next non-empty line
    if (values.empty()) {
        std::string line;
        while (std::getline(infile, line)) {
            if (!isIgnoredLine(line)) {
                std::istringstream newIss(line);
                while (newIss >> token) {
                    values.push_back(convertToValueType<ValueType>(token));
                }
                break;
            }
        }
    }
    return values;
}

