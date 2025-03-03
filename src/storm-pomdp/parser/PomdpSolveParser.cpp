#include "PomdpSolveParser.h"
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "Utility.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/api/builder.h"
#include "storm/api/export.h"
#include "storm/utility/logging.h"

namespace storm {
namespace pomdp {
namespace parser {

template<typename ValueType>
struct POMDPcomponents {
    std::vector<std::string> states;
    std::vector<std::string> actions;
    std::vector<std::string> observations;
    std::vector<ValueType> start_probabilities;
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> transitions;
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> observations_prob;
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> rewards;
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> newRewards;
    std::vector<std::string> newStates;  // Observation-based states
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> newTransitions;
    ValueType discount = storm::utility::one<ValueType>();
};

enum class MatrixType { NORMAL, IDENTITY, UNIFORM };

MatrixType getMatrixType(const std::string& line) {
    std::string trimmed = trim(line);
    if (trimmed == "identity") {
        return MatrixType::IDENTITY;
    } else if (trimmed == "uniform") {
        return MatrixType::UNIFORM;
    }
    return MatrixType::NORMAL;
}

template<typename ValueType>
std::vector<ValueType> generateIdentityRow(size_t size, size_t rowIndex) {
    std::vector<ValueType> row(size, storm::utility::zero<ValueType>());
    if (rowIndex < size) {
        row[rowIndex] = storm::utility::one<ValueType>();
    }
    return row;
}

template<typename ValueType>
std::vector<ValueType> generateUniformRow(size_t size) {
    ValueType uniformProb = storm::utility::one<ValueType>() / static_cast<ValueType>(size);
    return std::vector<ValueType>(size, uniformProb);
}

template<typename ValueType>
POMDPcomponents<ValueType> parsePomdpFile(const std::string& filename) {
    POMDPcomponents<ValueType> pomdp;
    std::ifstream infile(filename);
    if (!infile.good()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return pomdp;
    }
    std::string line;
    STORM_PRINT("Starting to parse file");

    while (std::getline(infile, line)) {
        if (isIgnoredLine(line))
            continue;

        line = trim(line);
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word.empty())
            continue;

        if (word == "discount:") {
            std::string token;
            iss >> token;
            pomdp.discount = convertToValueType<ValueType>(token);
        } else if (word == "values:") {
            continue;
        } else if (word == "states:") {
            parseIdentifierVector(iss, pomdp.states, "state");
        } else if (word == "actions:") {
            parseIdentifierVector(iss, pomdp.actions, "action");
        } else if (word == "observations:") {
            STORM_PRINT("Parsing observations");
            parseIdentifierVector(iss, pomdp.observations, "observation");
        } else if (word == "start:") {
            STORM_PRINT("Parsing start probabilities");
            std::vector<ValueType> startProbs = readArrayTokens<ValueType>(iss, infile);
            pomdp.start_probabilities.insert(pomdp.start_probabilities.end(), startProbs.begin(), startProbs.end());
        } else if (word == "T:") {
            STORM_PRINT("Parsing transition(s)");

            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            int colonCount = std::count(rest.begin(), rest.end(), ':');
            if (colonCount >= 2) {
                // <action> : <start-state> : <end-state> <probability>
                std::istringstream restStream(rest);
                std::string action, start_state, remainder;
                std::getline(restStream, action, ':');
                action = trim(action);
                std::getline(restStream, start_state, ':');
                start_state = trim(start_state);
                std::getline(restStream, remainder);
                remainder = trim(remainder);
                // <end_state> <probability>
                std::istringstream remainderStream(remainder);
                std::string end_state;
                remainderStream >> end_state;
                end_state = trim(end_state);

                std::string probability_token;
                ValueType probability;
                if (remainderStream >> probability_token) {
                    probability = convertToValueType<ValueType>(probability_token);
                } else {
                    // try next line
                    std::string nextLine;
                    if (std::getline(infile, nextLine) && !isIgnoredLine(nextLine)) {
                        std::istringstream nextLineStream(trim(nextLine));
                        if (nextLineStream >> probability_token) {
                            probability = convertToValueType<ValueType>(probability_token);
                        } else {
                            STORM_PRINT("Error: Invalid probability format for T: " + action + " : " + start_state + " : " + end_state + "\n");
                            continue;
                        }
                    } else {
                        STORM_PRINT("Error: Missing probability value for T: " + action + " : " + start_state + " : " + end_state + "\n");
                        continue;
                    }
                }

                if (probability != storm::utility::zero<ValueType>()) {
                    if (action == "*") {
                        for (const auto& act : pomdp.actions) {
                            pomdp.transitions[act][start_state + ":" + end_state] = probability;
                        }
                    } else if (start_state == "*") {
                        for (const auto& state : pomdp.states) {
                            pomdp.transitions[action][state + ":" + end_state] = probability;
                        }
                    } else {
                        pomdp.transitions[action][start_state + ":" + end_state] = probability;
                    }
                }
            } else if (colonCount == 1) {
                // <action> : <start-state> <p1> <p2> ...
                std::istringstream restStream(rest);
                std::string action, start_state;
                std::getline(restStream, action, ':');
                action = trim(action);
                restStream >> start_state;
                start_state = trim(start_state);

                std::vector<ValueType> probabilities = readArrayTokens<ValueType>(restStream, infile);
                for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                    if (probabilities[i] != storm::utility::zero<ValueType>()) {
                        if (action == "*") {
                            for (const auto& act : pomdp.actions) {
                                pomdp.transitions[act][start_state + ":" + pomdp.states[i]] = probabilities[i];
                            }
                        } else {
                            pomdp.transitions[action][start_state + ":" + pomdp.states[i]] = probabilities[i];
                        }
                    }
                }
            } else {
                // T: <action>
                std::string action = trim(rest);

                std::string nextLine;
                std::getline(infile, nextLine);
                if (isIgnoredLine(nextLine)) {
                    while (std::getline(infile, nextLine) && isIgnoredLine(nextLine)) {
                    }
                }

                MatrixType matrixType = getMatrixType(nextLine);

                if (matrixType == MatrixType::IDENTITY) {
                    for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            ValueType prob = (i == j) ? storm::utility::one<ValueType>() : storm::utility::zero<ValueType>();
                            if (prob != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) {
                                        pomdp.transitions[act][pomdp.states[i] + ":" + pomdp.states[j]] = prob;
                                    }
                                } else {
                                    pomdp.transitions[action][pomdp.states[i] + ":" + pomdp.states[j]] = prob;
                                }
                            }
                        }
                    }
                } else if (matrixType == MatrixType::UNIFORM) {
                    ValueType uniformProb = storm::utility::one<ValueType>() / static_cast<ValueType>(pomdp.states.size());
                    for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.transitions[act][pomdp.states[i] + ":" + pomdp.states[j]] = uniformProb;
                                }
                            } else {
                                pomdp.transitions[action][pomdp.states[i] + ":" + pomdp.states[j]] = uniformProb;
                            }
                        }
                    }
                } else {
                    std::istringstream rowStream(nextLine);
                    for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                        std::string token;
                        rowStream >> token;
                        ValueType prob = convertToValueType<ValueType>(token);
                        if (prob != storm::utility::zero<ValueType>()) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.transitions[act][pomdp.states[0] + ":" + pomdp.states[j]] = prob;
                                }
                            } else {
                                pomdp.transitions[action][pomdp.states[0] + ":" + pomdp.states[j]] = prob;
                            }
                        }
                    }

                    for (uint64_t i = 1; i < pomdp.states.size(); i++) {
                        std::getline(infile, line);
                        if (isIgnoredLine(line)) {
                            i--;
                            continue;
                        }
                        std::istringstream rowStream(line);
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            std::string token;
                            rowStream >> token;
                            ValueType prob = convertToValueType<ValueType>(token);
                            if (prob != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) {
                                        pomdp.transitions[act][pomdp.states[i] + ":" + pomdp.states[j]] = prob;
                                    }
                                } else {
                                    pomdp.transitions[action][pomdp.states[i] + ":" + pomdp.states[j]] = prob;
                                }
                            }
                        }
                    }
                }
            }
        } else if (word == "O:") {
            STORM_PRINT("Parsing observation probabilities \n");

            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            int colonCount = std::count(rest.begin(), rest.end(), ':');
            if (colonCount >= 2) {
                // <action> : <end-state> : <observation> <probability>
                std::istringstream restStream(rest);
                std::string action, end_state, remainder;
                std::getline(restStream, action, ':');
                action = trim(action);
                std::getline(restStream, end_state, ':');
                end_state = trim(end_state);
                std::getline(restStream, remainder);
                remainder = trim(remainder);
                std::istringstream remainderStream(remainder);
                std::string observation;
                remainderStream >> observation;
                observation = trim(observation);

                std::string probability_token;
                ValueType probability;
                if (remainderStream >> probability_token) {
                    probability = convertToValueType<ValueType>(probability_token);
                } else {
                    // try next line
                    std::string nextLine;
                    if (std::getline(infile, nextLine) && !isIgnoredLine(nextLine)) {
                        std::istringstream nextLineStream(trim(nextLine));
                        if (nextLineStream >> probability_token) {
                            probability = convertToValueType<ValueType>(probability_token);
                        } else {
                            STORM_PRINT("Error: Invalid probability format for O: " + action + " : " + end_state + " : " + observation + "\n");
                            continue;
                        }
                    } else {
                        STORM_PRINT("Error: Missing probability value for O: " + action + " : " + end_state + " : " + observation + "\n");
                        continue;
                    }
                }

                if (probability != storm::utility::zero<ValueType>()) {
                    if (action == "*" && end_state == "*") {
                        for (const auto& act : pomdp.actions) {
                            for (const auto& state : pomdp.states) {
                                pomdp.observations_prob[act][state + ":" + observation] = probability;
                            }
                        }
                    } else if (action == "*") {
                        for (const auto& act : pomdp.actions) {
                            pomdp.observations_prob[act][end_state + ":" + observation] = probability;
                        }
                    } else if (end_state == "*") {
                        for (const auto& state : pomdp.states) {
                            pomdp.observations_prob[action][state + ":" + observation] = probability;
                        }
                    } else {
                        pomdp.observations_prob[action][end_state + ":" + observation] = probability;
                    }
                }
            } else if (rest.find(':') != std::string::npos) {
                // <action> : <end-state> <p1> <p2> ... <pN>
                std::istringstream restStream(rest);
                std::string action, end_state;
                std::getline(restStream, action, ':');
                action = trim(action);
                restStream >> end_state;
                end_state = trim(end_state);
                std::vector<ValueType> probabilities = readArrayTokens<ValueType>(restStream, infile);

                for (uint64_t i = 0; i < pomdp.observations.size(); i++) {
                    if (probabilities[i] != storm::utility::zero<ValueType>()) {
                        if (action == "*") {
                            for (const auto& act : pomdp.actions) {
                                pomdp.observations_prob[act][end_state + ":" + pomdp.observations[i]] = probabilities[i];
                            }
                        } else {
                            pomdp.observations_prob[action][end_state + ":" + pomdp.observations[i]] = probabilities[i];
                        }
                    }
                }
            } else {
                // O: <action>
                std::string action = trim(rest);
                std::string nextLine;
                std::getline(infile, nextLine);
                if (isIgnoredLine(nextLine)) {
                    while (std::getline(infile, nextLine) && isIgnoredLine(nextLine)) {
                    }
                }

                MatrixType matrixType = getMatrixType(nextLine);

                if (matrixType == MatrixType::IDENTITY) {
                    if (pomdp.states.size() == pomdp.observations.size()) {
                        for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                            for (uint64_t j = 0; j < pomdp.observations.size(); j++) {
                                ValueType prob = (i == j) ? storm::utility::one<ValueType>() : storm::utility::zero<ValueType>();
                                if (prob != storm::utility::zero<ValueType>()) {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) {
                                            pomdp.observations_prob[act][pomdp.states[i] + ":" + pomdp.observations[j]] = prob;
                                        }
                                    } else {
                                        pomdp.observations_prob[action][pomdp.states[i] + ":" + pomdp.observations[j]] = prob;
                                    }
                                }
                            }
                        }
                    } else {
                        STORM_PRINT("Error: Identity matrix for observations requires equal number of states and observations.");
                        continue;
                    }
                } else if (matrixType == MatrixType::UNIFORM) {
                    ValueType uniformProb = storm::utility::one<ValueType>() / static_cast<ValueType>(pomdp.observations.size());
                    for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                        for (uint64_t j = 0; j < pomdp.observations.size(); j++) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.observations_prob[act][pomdp.states[i] + ":" + pomdp.observations[j]] = uniformProb;
                                }
                            } else {
                                pomdp.observations_prob[action][pomdp.states[i] + ":" + pomdp.observations[j]] = uniformProb;
                            }
                        }
                    }
                } else if (matrixType == MatrixType::NORMAL) {
                    std::istringstream rowStream(nextLine);
                    for (uint64_t j = 0; j < pomdp.observations.size(); j++) {
                        std::string token;
                        rowStream >> token;
                        ValueType prob = convertToValueType<ValueType>(token);
                        if (prob != storm::utility::zero<ValueType>()) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.observations_prob[act][pomdp.states[0] + ":" + pomdp.observations[j]] = prob;
                                }
                            } else {
                                pomdp.observations_prob[action][pomdp.states[0] + ":" + pomdp.observations[j]] = prob;
                            }
                        }
                    }

                    for (uint64_t i = 1; i < pomdp.states.size(); i++) {
                        std::getline(infile, line);
                        if (isIgnoredLine(line)) {
                            i--;
                            continue;
                        }
                        std::istringstream rowStream(line);
                        for (uint64_t j = 0; j < pomdp.observations.size(); j++) {
                            std::string token;
                            rowStream >> token;
                            ValueType prob = convertToValueType<ValueType>(token);
                            if (prob != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) {
                                        pomdp.observations_prob[act][pomdp.states[i] + ":" + pomdp.observations[j]] = prob;
                                    }
                                } else {
                                    pomdp.observations_prob[action][pomdp.states[i] + ":" + pomdp.observations[j]] = prob;
                                }
                            }
                        }
                    }
                }
            }
        } else if (word == "R:") {
            STORM_PRINT("Parsing rewards");
            std::string headerLine;
            std::getline(iss, headerLine);
            headerLine = trim(headerLine);
            size_t colonCount = std::count(headerLine.begin(), headerLine.end(), ':');

            // <action> : <start-state> : <end-state> : <observation> <reward_value>
            if (colonCount >= 3) {
                std::istringstream headerStream(headerLine);
                std::string action, start_state, end_state, remainder;
                std::getline(headerStream, action, ':');
                action = trim(action);
                std::getline(headerStream, start_state, ':');
                start_state = trim(start_state);
                std::getline(headerStream, end_state, ':');
                end_state = trim(end_state);
                std::getline(headerStream, remainder);
                remainder = trim(remainder);
                // <observation> <reward_value>
                std::istringstream remainderStream(remainder);
                std::string observation;
                remainderStream >> observation;
                observation = trim(observation);

                std::string rewardStr;
                ValueType rewardVal;
                if (remainderStream >> rewardStr) {
                    rewardVal = convertToValueType<ValueType>(rewardStr);
                } else {
                    //try next line
                    std::string nextLine;
                    if (std::getline(infile, nextLine) && !isIgnoredLine(nextLine)) {
                        std::istringstream nextLineStream(trim(nextLine));
                        if (nextLineStream >> rewardStr) {
                            rewardVal = convertToValueType<ValueType>(rewardStr);
                        } else {
                            STORM_PRINT("Error: Invalid reward format for R: " + action + " : " + start_state + " : " + end_state + " : " + observation + "\n");
                            continue;
                        }
                    } else {
                        STORM_PRINT("Error: Missing reward value for R: " + action + " : " + start_state + " : " + end_state + " : " + observation + "\n");
                        continue;
                    }
                }

                if (rewardVal != storm::utility::zero<ValueType>()) {
                    std::vector<std::string> actionsToUse;
                    if (action == "*")
                        actionsToUse = pomdp.actions;
                    else
                        actionsToUse.push_back(action);

                    std::vector<std::string> startStatesToUse;
                    if (start_state == "*")
                        startStatesToUse = pomdp.states;
                    else
                        startStatesToUse.push_back(start_state);

                    std::vector<std::string> endStatesToUse;
                    if (end_state == "*")
                        endStatesToUse = pomdp.states;
                    else
                        endStatesToUse.push_back(end_state);

                    std::vector<std::string> observationsToUse;
                    if (observation == "*")
                        observationsToUse = pomdp.observations;
                    else
                        observationsToUse.push_back(observation);

                    for (const auto& ss : startStatesToUse) {
                        for (const auto& act : actionsToUse) {
                            for (const auto& es : endStatesToUse) {
                                for (const auto& obs : observationsToUse) {
                                    pomdp.rewards[ss][act + ":" + es + ":" + obs] = rewardVal;
                                    STORM_PRINT("Created reward for key: " + act + ":" + es + ":" + obs + " under start state: " + ss + "\n");
                                }
                            }
                        }
                    }
                }
            }
            // R: <action> : <start-state> : <end-state> <token> <token> ... <token>
            else if (colonCount == 2) {
                std::istringstream headerStream(headerLine);
                std::string action, start_state, end_state;
                std::getline(headerStream, action, ':');
                std::getline(headerStream, start_state, ':');
                std::getline(headerStream, end_state);
                action = trim(action);
                start_state = trim(start_state);
                end_state = trim(end_state);
                std::vector<ValueType> rewardsVec = readArrayTokens<ValueType>(headerStream, infile);

                if (start_state == "*") {
                    for (const auto& s : pomdp.states) {
                        if (end_state == "*") {
                            for (size_t i = 0; i < pomdp.states.size() && i < rewardsVec.size(); ++i) {
                                if (rewardsVec[i] != storm::utility::zero<ValueType>()) {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + pomdp.states[i]] = rewardsVec[i];
                                    } else {
                                        pomdp.rewards[s][action + ":" + pomdp.states[i]] = rewardsVec[i];
                                    }
                                }
                            }
                        } else {
                            if (!rewardsVec.empty() && rewardsVec[0] != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + end_state] = rewardsVec[0];
                                } else {
                                    pomdp.rewards[s][action + ":" + end_state] = rewardsVec[0];
                                }
                            }
                        }
                    }
                } else {
                    if (end_state == "*") {
                        for (size_t i = 0; i < pomdp.states.size() && i < rewardsVec.size(); ++i) {
                            if (rewardsVec[i] != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + pomdp.states[i]] = rewardsVec[i];
                                } else {
                                    pomdp.rewards[start_state][action + ":" + pomdp.states[i]] = rewardsVec[i];
                                }
                            }
                        }
                    } else {
                        if (!rewardsVec.empty() && rewardsVec[0] != storm::utility::zero<ValueType>()) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + end_state] = rewardsVec[0];
                            } else {
                                pomdp.rewards[start_state][action + ":" + end_state] = rewardsVec[0];
                            }
                        }
                    }
                }
            } else if (colonCount == 1) {
                std::istringstream headerStream(headerLine);
                std::string action, start_state;
                std::getline(headerStream, action, ':');
                action = trim(action);
                std::getline(headerStream, start_state);
                start_state = trim(start_state);

                std::string nextLine;
                std::getline(infile, nextLine);
                if (isIgnoredLine(nextLine)) {
                    while (std::getline(infile, nextLine) && isIgnoredLine(nextLine)) {
                    }
                }

                MatrixType matrixType = getMatrixType(nextLine);

                if (matrixType == MatrixType::IDENTITY) {
                    for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            ValueType rewardValue = (i == j) ? storm::utility::one<ValueType>() : storm::utility::zero<ValueType>();
                            if (rewardValue != storm::utility::zero<ValueType>()) {
                                if (start_state == "*") {
                                    for (const auto& s : pomdp.states) {
                                        if (action == "*") {
                                            for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + pomdp.states[j]] = rewardValue;
                                        } else {
                                            pomdp.rewards[s][action + ":" + pomdp.states[j]] = rewardValue;
                                        }
                                    }
                                } else {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + pomdp.states[j]] = rewardValue;
                                    } else {
                                        pomdp.rewards[start_state][action + ":" + pomdp.states[j]] = rewardValue;
                                    }
                                }
                            }
                        }
                    }
                } else if (matrixType == MatrixType::UNIFORM) {
                    ValueType uniformReward = storm::utility::one<ValueType>() / static_cast<ValueType>(pomdp.states.size());
                    for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            if (start_state == "*") {
                                for (const auto& s : pomdp.states) {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + pomdp.states[j]] = uniformReward;
                                    } else {
                                        pomdp.rewards[s][action + ":" + pomdp.states[j]] = uniformReward;
                                    }
                                }
                            } else {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + pomdp.states[j]] = uniformReward;
                                } else {
                                    pomdp.rewards[start_state][action + ":" + pomdp.states[j]] = uniformReward;
                                }
                            }
                        }
                    }
                } else {
                    std::istringstream firstRowStream(nextLine);
                    if (start_state == "*") {
                        for (const auto& s : pomdp.states) {
                            for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                                std::string token;
                                firstRowStream >> token;
                                ValueType val = convertToValueType<ValueType>(token);
                                if (val != storm::utility::zero<ValueType>()) {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + pomdp.states[j]] = val;
                                    } else {
                                        pomdp.rewards[s][action + ":" + pomdp.states[j]] = val;
                                    }
                                }
                            }

                            for (uint64_t i = 1; i < pomdp.states.size(); i++) {
                                std::string matrixLine;
                                std::getline(infile, matrixLine);
                                if (isIgnoredLine(matrixLine)) {
                                    i--;
                                    continue;
                                }
                                std::istringstream rowStream(matrixLine);
                                for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                                    std::string token;
                                    rowStream >> token;
                                    ValueType val = convertToValueType<ValueType>(token);
                                    if (val != storm::utility::zero<ValueType>()) {
                                        if (action == "*") {
                                            for (const auto& act : pomdp.actions) pomdp.rewards[s][act + ":" + pomdp.states[j]] = val;
                                        } else {
                                            pomdp.rewards[s][action + ":" + pomdp.states[j]] = val;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                            std::string token;
                            firstRowStream >> token;
                            ValueType val = convertToValueType<ValueType>(token);
                            if (val != storm::utility::zero<ValueType>()) {
                                if (action == "*") {
                                    for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + pomdp.states[j]] = val;
                                } else {
                                    pomdp.rewards[start_state][action + ":" + pomdp.states[j]] = val;
                                }
                            }
                        }
                        for (uint64_t i = 1; i < pomdp.states.size(); i++) {
                            std::string matrixLine;
                            std::getline(infile, matrixLine);
                            if (isIgnoredLine(matrixLine)) {
                                i--;
                                continue;
                            }
                            std::istringstream rowStream(matrixLine);
                            for (uint64_t j = 0; j < pomdp.states.size(); j++) {
                                std::string token;
                                rowStream >> token;
                                ValueType val = convertToValueType<ValueType>(token);
                                if (val != storm::utility::zero<ValueType>()) {
                                    if (action == "*") {
                                        for (const auto& act : pomdp.actions) pomdp.rewards[start_state][act + ":" + pomdp.states[j]] = val;
                                    } else {
                                        pomdp.rewards[start_state][action + ":" + pomdp.states[j]] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return pomdp;
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> buildTransitionMatrix(const std::unordered_map<std::string, std::unordered_map<std::string, ValueType>>& newTransitions,
                                                              const std::vector<std::string>& newStates, const std::vector<std::string>& actions) {
    STORM_PRINT("Building transition matrix");

    uint64_t numChoices = 0;
    for (const auto& state : newStates) {
        if (state == "initial")
            numChoices += 1;
        else
            numChoices += (actions.size() - 1);
    }

    uint64_t numStates = newStates.size();
    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : newStates) {
        stateIndices[state] = index++;
    }

    uint64_t entryCount = 0;
    for (const auto& [startState, actionMap] : newTransitions) {
        entryCount += actionMap.size();
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(numChoices,  // rows
                                                           numStates,   // columns
                                                           entryCount,  // upper bound nonzero entries
                                                           true,        // forceDimensions
                                                           true,        // hasCustomRowGrouping
                                                           numStates);  // number of row groups

    uint64_t rowIndex = 0;
    for (const auto& state : newStates) {
        uint64_t rowGroupStart = rowIndex;
        builder.newRowGroup(rowGroupStart);

        if (state == "initial") {
            if (newTransitions.find(state) != newTransitions.end()) {
                const auto& actionMap = newTransitions.at(state);
                for (const auto& [actionEndState, probability] : actionMap) {
                    size_t pos = actionEndState.find(':');
                    if (pos == std::string::npos)
                        throw std::runtime_error("Invalid action:endState format: " + actionEndState);
                    std::string transitionAction = actionEndState.substr(0, pos);
                    if (transitionAction == "start") {
                        std::string endState = actionEndState.substr(pos + 1);
                        if (stateIndices.find(endState) == stateIndices.end())
                            throw std::runtime_error("EndState not found in newStates: " + endState);
                        uint64_t colIndex = stateIndices[endState];
                        builder.addNextValue(rowIndex, colIndex, probability);
                    }
                }
            }
            rowIndex++;
        } else {
            for (const auto& action : actions) {
                if (action == "start")
                    continue;
                if (newTransitions.find(state) != newTransitions.end()) {
                    const auto& actionMap = newTransitions.at(state);
                    for (const auto& [actionEndState, probability] : actionMap) {
                        size_t pos = actionEndState.find(':');
                        if (pos == std::string::npos)
                            throw std::runtime_error("Invalid action:endState format: " + actionEndState);
                        std::string transitionAction = actionEndState.substr(0, pos);
                        if (transitionAction == action) {
                            std::string endState = actionEndState.substr(pos + 1);
                            if (stateIndices.find(endState) == stateIndices.end())
                                throw std::runtime_error("EndState not found in newStates: " + endState);
                            uint64_t colIndex = stateIndices[endState];
                            builder.addNextValue(rowIndex, colIndex, probability);
                        }
                    }
                }
                rowIndex++;
            }
        }
    }

    STORM_PRINT("Transition matrix built");
    return builder.build();
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> buildRewardMatrix(const std::unordered_map<std::string, std::unordered_map<std::string, ValueType>>& rewards,
                                                          const std::vector<std::string>& newStates, const std::vector<std::string>& actions) {
    STORM_PRINT("Building reward matrix");

    uint64_t numChoices = 0;
    for (const auto& state : newStates) {
        if (state == "initial")
            numChoices += 1;
        else
            numChoices += (actions.size() - 1);
    }

    uint64_t numStates = newStates.size();
    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : newStates) {
        stateIndices[state] = index++;
    }

    uint64_t entryCount = 0;
    for (const auto& [source, rewardMap] : rewards) {
        entryCount += rewardMap.size();
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(numChoices,  // rows
                                                           numStates,   // columns
                                                           entryCount,  // upper bound nonzero entries
                                                           true,        // forceDimensions
                                                           true,        // hasCustomRowGrouping
                                                           numStates);  // row groups

    uint64_t rowIndex = 0;
    for (const auto& state : newStates) {
        uint64_t rowGroupStart = rowIndex;
        builder.newRowGroup(rowGroupStart);

        if (state == "initial") {
            if (rewards.find(state) != rewards.end()) {
                const auto& rewardMap = rewards.at(state);
                for (const auto& [rewardKey, rewardValue] : rewardMap) {
                    size_t pos = rewardKey.find(':');
                    if (pos == std::string::npos)
                        throw std::runtime_error("Invalid reward key format: " + rewardKey);
                    std::string rewardAction = rewardKey.substr(0, pos);
                    if (rewardAction == "start") {
                        std::string targetState = rewardKey.substr(pos + 1);
                        if (stateIndices.find(targetState) == stateIndices.end())
                            throw std::runtime_error("Target state not found in newStates: " + targetState);
                        uint64_t colIndex = stateIndices[targetState];
                        builder.addNextValue(rowIndex, colIndex, rewardValue);
                    }
                }
            }
            rowIndex++;
        } else {
            for (const auto& action : actions) {
                if (action == "start")
                    continue;
                if (rewards.find(state) != rewards.end()) {
                    const auto& rewardMap = rewards.at(state);
                    for (const auto& [rewardKey, rewardValue] : rewardMap) {
                        size_t pos = rewardKey.find(':');
                        if (pos == std::string::npos)
                            throw std::runtime_error("Invalid reward key format: " + rewardKey);
                        std::string rewardAction = rewardKey.substr(0, pos);
                        if (rewardAction == action) {
                            std::string targetState = rewardKey.substr(pos + 1);
                            if (stateIndices.find(targetState) == stateIndices.end())
                                throw std::runtime_error("Target state not found in newStates: " + targetState);
                            uint64_t colIndex = stateIndices[targetState];
                            builder.addNextValue(rowIndex, colIndex, rewardValue);
                        }
                    }
                }
                rowIndex++;
            }
        }
    }

    STORM_PRINT("Reward matrix built");
    return builder.build();
}

template<typename ValueType>
std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> createNewTransitions(POMDPcomponents<ValueType>& pomdp) {
    STORM_PRINT("Creating new observation-based transitions");
    pomdp.newTransitions.clear();
    std::unordered_set<std::string> newStatesSet;

    for (const auto& action_entry : pomdp.transitions) {
        const std::string& action = action_entry.first;

        auto obsProbIt = pomdp.observations_prob.find(action);
        if (obsProbIt == pomdp.observations_prob.end()) {
            STORM_PRINT("No observation probabilities for action: " + action + "\n");
            continue;
        }
        const auto& obsProbMap = obsProbIt->second;

        for (const auto& transition_entry : action_entry.second) {
            std::string baseTransitionKey = transition_entry.first;
            size_t pos = baseTransitionKey.find(':');
            if (pos == std::string::npos)
                continue;
            std::string baseSource = baseTransitionKey.substr(0, pos);
            std::string baseDest = baseTransitionKey.substr(pos + 1);
            ValueType baseProb = transition_entry.second;

            std::string destPrefix = baseDest + ":";

            for (const auto& obsEntry : obsProbMap) {
                if (obsEntry.first.rfind(destPrefix, 0) == 0) {
                    std::string observation = obsEntry.first.substr(destPrefix.size());
                    ValueType obsProb = obsEntry.second;
                    ValueType newProb = baseProb * obsProb;
                    if (newProb > storm::utility::zero<ValueType>()) {
                        std::string newKey = action + ":" + baseDest + ":" + observation;
                        pomdp.newTransitions[baseSource][newKey] += newProb;
                        STORM_PRINT("Added new transition: " + baseSource + " -> " + newKey + " with probability \n");
                        newStatesSet.insert(baseDest + ":" + observation);
                    }
                }
            }
        }
    }

    for (const auto& state : newStatesSet) {
        pomdp.newStates.push_back(state);
        STORM_PRINT("Added new state: " + state + "\n");
    }

    return pomdp.newTransitions;
}

template<typename ValueType>
void addInitialStateTransitions(POMDPcomponents<ValueType>& pomdp_component) {
    const std::string initialState = "initial";
    const std::string action = "start";

    if (pomdp_component.start_probabilities.size() != pomdp_component.states.size()) {
        std::cerr << "Error: start_probabilities and states vectors must be of the same size." << std::endl;
        return;
    }

    for (uint64_t i = 0; i < pomdp_component.start_probabilities.size(); ++i) {
        ValueType probability = pomdp_component.start_probabilities[i];
        const std::string& state = pomdp_component.states[i];

        if (probability > storm::utility::zero<ValueType>()) {
            std::string newState = state + ":start";
            std::string actionEndStateKey = action + ":" + newState;
            pomdp_component.newTransitions[initialState][actionEndStateKey] = probability;
            pomdp_component.newStates.push_back(newState);
        }
    }
    pomdp_component.actions.push_back(action);
    pomdp_component.newStates.push_back(initialState);
}

template<typename ValueType>
void expandNewStatesTransitions(POMDPcomponents<ValueType>& pomdp_components) {
    STORM_PRINT("Expanding transitions to observation-based new states\n");

    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> expandedTransitions;

    for (const auto& entry : pomdp_components.newTransitions) {
        const std::string& startState = entry.first;
        const auto& transitionsForState = entry.second;

        if (startState == "initial") {
            for (const auto& t : transitionsForState) {
                STORM_PRINT("Adding transition: " + startState + " -> " + t.first + " with probability \n");
                expandedTransitions[startState][t.first] += t.second;
            }
        } else if (startState.find(':') != std::string::npos) {
            for (const auto& t : transitionsForState) {
                STORM_PRINT("Adding transition: " + startState + " -> " + t.first + " with probability \n");
                expandedTransitions[startState][t.first] += t.second;
            }
        } else {
            for (const auto& obsState : pomdp_components.newStates) {
                STORM_PRINT("Checking if " + obsState + " is a variant of " + startState + "\n");
                if (obsState.size() > startState.size() && obsState.compare(0, startState.size(), startState) == 0 && obsState[startState.size()] == ':') {
                    STORM_PRINT("Found variant: " + obsState + "\n");
                    for (const auto& t : transitionsForState) {
                        STORM_PRINT("Adding transition: " + obsState + " -> " + t.first + " with probability \n");
                        expandedTransitions[obsState][t.first] += t.second;
                    }
                }
            }
        }
    }

    pomdp_components.newTransitions = std::move(expandedTransitions);
}

template<typename ValueType>
void filterDuplicateEntries(std::unordered_map<std::string, std::unordered_map<std::string, ValueType>>& newTransitions) {
    for (auto& [startState, actionMap] : newTransitions) {
        std::unordered_map<std::string, ValueType> filteredActionMap;

        // Iterate through the inner map
        for (const auto& [actionEndState, probability] : actionMap) {
            if (filteredActionMap.find(actionEndState) == filteredActionMap.end()) {
                filteredActionMap[actionEndState] = probability;
            } else {
                std::cerr << "Duplicate entry removed: " << startState << " -> " << actionEndState << " (Probability: " << probability << ")\n";
            }
        }

        actionMap = std::move(filteredActionMap);
    }
}

template<typename ValueType>
std::vector<std::string> getObservationVariantsForBase(const POMDPcomponents<ValueType>& pomdp, const std::string& base) {
    std::vector<std::string> variants;
    for (const auto& ns : pomdp.newStates) {
        size_t pos = ns.find(':');
        if (pos != std::string::npos) {
            std::string nsBase = ns.substr(0, pos);
            if (nsBase == base) {
                variants.push_back(ns);
            }
        }
    }
    return variants;
}

template<typename ValueType>
void expandObservationBasedRewards(POMDPcomponents<ValueType>& pomdp) {
    STORM_PRINT("Expanding observation-based rewards");

    std::unordered_set<std::string> newStatesSet(pomdp.newStates.begin(), pomdp.newStates.end());

    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> newRewards;

    for (const auto& sourceEntry : pomdp.rewards) {
        std::string startKey = sourceEntry.first;
        const auto& innerMap = sourceEntry.second;

        std::vector<std::string> startStates;
        if (startKey.find(':') != std::string::npos) {
            startStates.push_back(startKey);
        } else {
            startStates = getObservationVariantsForBase(pomdp, startKey);
        }
        if (startStates.empty()) {
            STORM_PRINT("No observation variants found for start state: " + startKey + ". Skipping rewards for this key.");
            continue;
        }

        for (const auto& rewardEntry : innerMap) {
            std::string rewardKey = rewardEntry.first;
            size_t pos = rewardKey.find(':');
            if (pos == std::string::npos)
                continue;

            std::string actionPart = rewardKey.substr(0, pos);
            std::string targetKey = rewardKey.substr(pos + 1);

            std::vector<std::string> targetStates;
            if (targetKey.find(':') != std::string::npos) {
                targetStates.push_back(targetKey);
            } else {
                targetStates = getObservationVariantsForBase(pomdp, targetKey);
            }
            if (targetStates.empty()) {
                STORM_PRINT("No observation variants found for target state: " + targetKey + ". Skipping reward key: " + rewardKey + "\n");
                continue;
            }

            for (const auto& obsStart : startStates) {
                if (newStatesSet.find(obsStart) == newStatesSet.end()) {
                    STORM_PRINT("Skipping start state " + obsStart + " as it is not in newStatesSet.\n");
                    continue;
                }
                // Check if there is any transition from this observation-based start.
                if (pomdp.newTransitions.find(obsStart) == pomdp.newTransitions.end()) {
                    STORM_PRINT("No transitions from start state: " + obsStart + "\n");
                    continue;
                }
                const auto& transMap = pomdp.newTransitions.at(obsStart);
                for (const auto& obsTarget : targetStates) {
                    if (newStatesSet.find(obsTarget) == newStatesSet.end()) {
                        STORM_PRINT("Skipping target state " + obsTarget + " as it is not in newStatesSet.\n");
                        continue;
                    }
                    std::string transKey = actionPart + ":" + obsTarget;
                    if (transMap.find(transKey) != transMap.end()) {
                        newRewards[obsStart][transKey] = rewardEntry.second;
                        STORM_PRINT("Created reward for key: " + transKey + " under start state: " + obsStart + "\n");
                    } else {
                        STORM_PRINT("No corresponding transition found for reward key: " + transKey + " under start state: " + obsStart + "\n");
                    }
                }
            }
        }
    }

    pomdp.newRewards = newRewards;
}

template<typename ValueType>
PomdpSolveParserResult<ValueType> PomdpSolveParser<ValueType>::parsePomdpSolveFile(const std::string& filename) {
    POMDPcomponents<ValueType> pomdp = parsePomdpFile<ValueType>(filename);

    createNewTransitions(pomdp);
    addInitialStateTransitions(pomdp);
    expandNewStatesTransitions(pomdp);
    filterDuplicateEntries(pomdp.newTransitions);

    auto transitionMatrix = buildTransitionMatrix<ValueType>(pomdp.newTransitions, pomdp.newStates, pomdp.actions);

    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : pomdp.newStates) {
        stateIndices[state] = index++;
    }

    storm::models::sparse::StateLabeling stateLabeling(pomdp.newStates.size());
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", stateIndices["initial"]);

    expandObservationBasedRewards(pomdp);

    auto rewardMatrix = buildRewardMatrix<ValueType>(pomdp.newRewards, pomdp.newStates, pomdp.actions);

    std::optional<storm::storage::SparseMatrix<ValueType>> optRewardMatrix = std::make_optional(rewardMatrix);
    std::optional<std::vector<ValueType>> optStateRewardVec = std::nullopt;
    std::optional<std::vector<ValueType>> optStateActionRewardVec = std::nullopt;
    storm::models::sparse::StandardRewardModel<ValueType> rewardModel(optStateRewardVec, optStateActionRewardVec, optRewardMatrix);

    std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> rewardModels;
    rewardModels["default"] = std::move(rewardModel);

    auto new_pomdp = std::make_shared<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>(
        std::move(transitionMatrix), std::move(stateLabeling), std::move(rewardModels));

    if (std::find(pomdp.observations.begin(), pomdp.observations.end(), "start") == pomdp.observations.end()) {
        pomdp.observations.push_back("start");
    }
    if (std::find(pomdp.observations.begin(), pomdp.observations.end(), "initial") == pomdp.observations.end()) {
        pomdp.observations.push_back("initial");
    }
    std::vector<uint32_t> stateObservations(pomdp.newStates.size(), 0);
    for (uint64_t i = 0; i < pomdp.newStates.size(); ++i) {
        const std::string& state = pomdp.newStates[i];
        if (state == "initial") {
            auto it = std::find(pomdp.observations.begin(), pomdp.observations.end(), "initial");
            stateObservations[i] = static_cast<uint32_t>(std::distance(pomdp.observations.begin(), it));
        } else if (state.find(':') != std::string::npos) {
            std::string obsName = state.substr(state.find(':') + 1);
            auto it = std::find(pomdp.observations.begin(), pomdp.observations.end(), obsName);
            if (it != pomdp.observations.end()) {
                stateObservations[i] = static_cast<uint32_t>(std::distance(pomdp.observations.begin(), it));
            } else {
                pomdp.observations.push_back(obsName);
                stateObservations[i] = static_cast<uint32_t>(pomdp.observations.size() - 1);
            }
        }
    }
    new_pomdp->updateObservations(std::move(stateObservations), true);

    std::string outputFilePath = "/home/kaloyank/storm/kaloyanFork/storm/resources/examples/testfiles/parser/example.pomdp.dot";
    auto modelPtr = new_pomdp->template as<storm::models::sparse::Model<ValueType>>();
    storm::api::exportSparseModelAsDot(modelPtr, outputFilePath);

    ValueType discountFactor = static_cast<ValueType>(pomdp.discount);
    STORM_PRINT("End of parser!!!\n");
    PomdpSolveParserResult<ValueType> result;
    new_pomdp->setIsCanonic();
    result.pomdp = new_pomdp;
    result.discountFactor = discountFactor;
    return result;
}

template class PomdpSolveParser<double>;
template class PomdpSolveParser<storm::RationalNumber>;

}  // namespace parser
}  // namespace pomdp
}  // namespace storm
