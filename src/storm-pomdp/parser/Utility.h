#pragma once
#include <sstream>
#include <string>
#include <vector>

const std::vector<std::string> POMDP_KEYWORDS = {
    "discount", "values", "states", "actions", "observations", "start include", "start exclude", "start", "T", "O", "R",
};

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

// Helper function to trim whitespace from both ends of a string and remove comments
std::string trim(const std::string& str) {
    // Remove comments
    std::string result = str;
    size_t commentPos = result.find('#');
    if (commentPos != std::string::npos) {
        result = result.substr(0, commentPos);
    }
    
    // Trim whitespace
    size_t first = result.find_first_not_of(" \t\n\r");
    size_t last = result.find_last_not_of(" \t\n\r");
    if (first == std::string::npos || last == std::string::npos) {
        return "";
    }
    
    return result.substr(first, last - first + 1);
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

bool startsWithKeyword(const std::string& line) {
    std::string trimmed = trim(line);
    for (const auto& keyword : POMDP_KEYWORDS) {

        if (trimmed.find(keyword) == 0) {
            size_t pos = keyword.length();

            // Only whitespace or/and colon after keyword is allowed
            if (pos < trimmed.length() && (trimmed[pos] == ':' || std::isspace(trimmed[pos]))) {
                if (trimmed[pos] == ':') {
                    return true;
                }

                size_t colonPos = trimmed.find(':', pos);
                if (colonPos != std::string::npos) {
                    bool onlyWhitespace = true;
                    for (size_t i = pos; i < colonPos; i++) {
                        if (!std::isspace(trimmed[i])) {
                            onlyWhitespace = false;
                            break;
                        }
                    }
                    if (onlyWhitespace) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Parse identifiers that can span multiple lines
void parseIdentifierVector(std::istringstream& iss, std::vector<std::string>& vec, const std::string& identifierName, std::ifstream& infile) {
    std::string token;

    if (iss >> token) {
        trim(token);
        bool is_number = !token.empty() && std::all_of(token.begin(), token.end(), ::isdigit);

        if (is_number) {
            int count = std::stoi(token);
            for (int i = 0; i < count; i++) {
                vec.push_back(std::to_string(i));
            }
            return;
        }

        vec.push_back(trim(token));

        while (iss >> token) {
            vec.push_back(trim(token));
        }
    }

    std::string line;
    std::streampos prevPos = infile.tellg();

    while (std::getline(infile, line)) {
        if (isIgnoredLine(line)) {
            continue;
        }
        trim(line);

        if (startsWithKeyword(line)) {
            infile.seekg(prevPos);
            break;
        }

        std::istringstream lineStream(line);
        while (lineStream >> token) {
            vec.push_back(trim(token));
        }

        prevPos = infile.tellg();
    }
}

std::string extractKeyword(const std::string& line) {
    if (!startsWithKeyword(line)) {
        return "";
    }

    std::string trimmed = trim(line);

    for (const auto& keyword : POMDP_KEYWORDS) {
        if (trimmed.find(keyword) == 0) {
            return keyword;
        }
    }

    return "";
}

std::string extractAfterColon(const std::string& line) {
    std::string trimmed = trim(line);

    std::string keyword = "";
    for (const auto& k : POMDP_KEYWORDS) {
        if (trimmed.find(k) == 0) {
            keyword = k;
            break;
        }
    }

    if (keyword.empty()) {
        return "";
    }

    size_t startPos = keyword.length();
    size_t colonPos = trimmed.find(':', startPos);

    if (colonPos == std::string::npos) {
        return "";
    }

    return trim(trimmed.substr(colonPos + 1));
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
