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
    std::vector<std::string> newStates;  // Observation-incorporated states
    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> newTransitions;
    ValueType discount = storm::utility::one<ValueType>();
};

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
                std::string end_state, probability_token;
                remainderStream >> end_state >> probability_token;
                ValueType probability = convertToValueType<ValueType>(probability_token);
                end_state = trim(end_state);

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
                // <action> : <start-state> <p1> <p2> ... <pN>
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
                for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                    std::getline(infile, line);
                    if (isIgnoredLine(line))
                        continue;
                    std::istringstream rowStream(line);
                    for (int j = 0; j < pomdp.states.size(); j++) {
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
        } else if (word == "O:") {
            STORM_PRINT("Parsing observation probabilities");

            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            int colonCount = std::count(rest.begin(), rest.end(), ':');
            if (colonCount >= 2) {
                //<action> : <end-state> : <observation> <probability>
                std::istringstream restStream(rest);
                std::string action, end_state, remainder;
                std::getline(restStream, action, ':');
                action = trim(action);
                std::getline(restStream, end_state, ':');
                end_state = trim(end_state);
                std::getline(restStream, remainder);
                remainder = trim(remainder);
                //<observation> <probability>
                std::istringstream remainderStream(remainder);
                std::string observation, probability_token;
                remainderStream >> observation >> probability_token;
                ValueType probability = convertToValueType<ValueType>(probability_token);
                observation = trim(observation);

                if (probability != storm::utility::zero<ValueType>()) {
                    if (action == "*") {
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
                for (uint64_t i = 0; i < pomdp.states.size(); i++) {
                    std::getline(infile, line);
                    if (isIgnoredLine(line))
                        continue;
                    std::istringstream rowStream(line);
                    for (int j = 0; j < pomdp.observations.size(); j++) {
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
                std::string observation, rewardStr;
                remainderStream >> observation >> rewardStr;
                ValueType rewardVal = convertToValueType<ValueType>(rewardStr);
                observation = trim(observation);

                if (rewardVal != storm::utility::zero<ValueType>()) {
                    if (action == "*") {
                        for (const auto& act : pomdp.actions) {
                            pomdp.rewards[trim(start_state)][act + ":" + trim(end_state) + ":" + trim(observation)] = rewardVal;
                        }
                    } else if (start_state == "*") {
                        for (const auto& s : pomdp.states) {
                            pomdp.rewards[s][action + ":" + trim(end_state) + ":" + trim(observation)] = rewardVal;
                        }
                    } else if (end_state == "*") {
                        for (const auto& s : pomdp.states) {
                            pomdp.rewards[trim(start_state)][action + ":" + s + ":" + trim(observation)] = rewardVal;
                        }
                    } else if (observation == "*") {
                        for (const auto& obs : pomdp.observations) {
                            pomdp.rewards[trim(start_state)][action + ":" + trim(end_state) + ":" + obs] = rewardVal;
                        }
                    } else {
                        pomdp.rewards[trim(start_state)][action + ":" + trim(end_state) + ":" + trim(observation)] = rewardVal;
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
            }
            // <action> : <start-state>
            else if (colonCount == 1) {
                std::istringstream headerStream(headerLine);
                std::string action, start_state;
                std::getline(headerStream, action, ':');
                action = trim(action);
                std::getline(headerStream, start_state);
                start_state = trim(start_state);
                if (start_state == "*") {
                    for (const auto& s : pomdp.states) {
                        std::string matrixLine;
                        std::getline(infile, matrixLine);
                        std::istringstream rowStream(matrixLine);
                        for (size_t j = 0; j < pomdp.states.size(); ++j) {
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
                } else {
                    for (size_t i = 0; i < pomdp.states.size(); ++i) {
                        std::string matrixLine;
                        std::getline(infile, matrixLine);
                        std::istringstream rowStream(matrixLine);
                        for (size_t j = 0; j < pomdp.states.size(); ++j) {
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

    return pomdp;
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> buildTransitionMatrix(const std::unordered_map<std::string, std::unordered_map<std::string, ValueType>>& newTransitions,
                                                              const std::vector<std::string>& newStates, const std::vector<std::string>& actions) {
    STORM_PRINT("Building transition matrix");

    uint64_t numStates = newStates.size();
    uint64_t numChoices = newStates.size() * actions.size();

    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : newStates) {
        stateIndices[state] = index++;
    }

    uint64_t entryCount = 0;
    for (const auto& [startState, actionMap] : newTransitions) {
        entryCount += actionMap.size();
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(numChoices,  // number of rows (each valid state-action pair)
                                                           numStates,   // number of columns (states)
                                                           entryCount,  // upper bound on nonzero entries
                                                           true,        // forceDimensions
                                                           true,        // hasCustomRowGrouping
                                                           numStates    // number of row groups (one per state)
    );

    uint64_t rowIndex = 0;
    for (const auto& state : newStates) {
        uint64_t rowGroupStart = rowIndex;
        builder.newRowGroup(rowGroupStart);

        for (const auto& action : actions) {
            if (newTransitions.find(state) != newTransitions.end()) {
                const auto& actionMap = newTransitions.at(state);
                for (const auto& [actionEndState, probability] : actionMap) {
                    auto pos = actionEndState.find(':');
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

    STORM_PRINT("Transition matrix built");
    return builder.build();
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> buildRewardMatrix(const std::unordered_map<std::string, std::unordered_map<std::string, ValueType>>& rewards,
                                                          const std::vector<std::string>& newStates, const std::vector<std::string>& actions) {
    STORM_PRINT("Building reward matrix");

    uint64_t numStates = newStates.size();
    uint64_t numChoices = newStates.size() * actions.size();

    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : newStates) {
        stateIndices[state] = index++;
    }

    uint64_t entryCount = 0;
    for (const auto& [source, rewardMap] : rewards) {
        entryCount += rewardMap.size();
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(numChoices,  // number of rows (each valid state-action pair)
                                                           numStates,   // number of columns (states)
                                                           entryCount,  // upper bound on nonzero entries
                                                           true,        // forceDimensions
                                                           true,        // hasCustomRowGrouping
                                                           numStates    // number of row groups (one per state)
    );

    uint64_t rowIndex = 0;
    for (const auto& state : newStates) {
        uint64_t rowGroupStart = rowIndex;
        builder.newRowGroup(rowGroupStart);
        for (const auto& action : actions) {
            if (rewards.find(state) != rewards.end()) {
                const auto& rewardMap = rewards.at(state);
                for (const auto& [rewardKey, rewardValue] : rewardMap) {
                    auto pos = rewardKey.find(':');
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

    STORM_PRINT("Reward matrix built");
    return builder.build();
}

template<typename ValueType>
std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> createNewTransitions(POMDPcomponents<ValueType>& pomdp) {
    std::unordered_set<std::string> uniqueStates;
    STORM_PRINT("Line 300");
    for (const auto& action_entry : pomdp.transitions) {
        const std::string& action = action_entry.first;

        for (const auto& transition_entry : action_entry.second) {
            auto pos = transition_entry.first.find(':');
            std::string startState = transition_entry.first.substr(0, pos);
            std::string endState = transition_entry.first.substr(pos + 1);
            ValueType transition_prob = transition_entry.second;

            ValueType totalProbability = storm::utility::zero<ValueType>();

            if (pomdp.observations_prob.count(action) > 0) {
                std::string obs_key_prefix = endState + ":";
                for (const auto& obs_entry : pomdp.observations_prob.at(action)) {
                    if (obs_entry.first.rfind(obs_key_prefix, 0) == 0) {
                        std::string observation = obs_entry.first.substr(obs_key_prefix.length());
                        ValueType obs_prob = obs_entry.second;

                        ValueType new_prob = transition_prob * obs_prob;
                        std::string newEndState = endState + ":" + observation;

                        if (new_prob > storm::utility::zero<ValueType>()) {
                            std::string actionEndStateKey = action + ":" + newEndState;
                            pomdp.newTransitions[startState][actionEndStateKey] = new_prob;
                            totalProbability += new_prob;
                            uniqueStates.insert(newEndState);
                            STORM_PRINT(actionEndStateKey);
                        }
                    }
                }
            }

            if (totalProbability < transition_prob) {
                ValueType remaining_prob = transition_prob - totalProbability;
                std::string actionEndStateKey = action + ":" + endState;
                pomdp.newTransitions[startState][actionEndStateKey] = remaining_prob;
            }
        }
    }

    pomdp.newStates.assign(uniqueStates.begin(), uniqueStates.end());
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
    STORM_PRINT("Line 368 - Expanding transitions to new states");
    // For each new state that is derived from an original state by appending an observation,
    // copy all transitions from the original state to the new state.
    for (const auto& newState : pomdp_components.newStates) {
        auto pos = newState.find(':');
        if (pos == std::string::npos)
            continue;

        // Extract the original state name
        std::string originalState = newState.substr(0, pos);

        // If the original state exists in the newTransitions mapping, copy its transitions.
        if (pomdp_components.newTransitions.find(originalState) != pomdp_components.newTransitions.end()) {
            for (const auto& actionEntry : pomdp_components.newTransitions[originalState]) {
                // Merge the transition probability (if newState already has a transition for the same key,
                // we add the probability; otherwise, the operator[] default constructs it to 0)
                pomdp_components.newTransitions[newState][actionEntry.first] += actionEntry.second;
            }
        }
    }
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
void expandObservationBasedRewards(POMDPcomponents<ValueType>& pomdp) {
    STORM_PRINT("Expanding observation-based rewards");

    std::unordered_map<std::string, std::vector<std::string>> obsVariants;
    for (const auto& ns : pomdp.newStates) {
        size_t pos = ns.find(':');
        if (pos != std::string::npos) {
            std::string orig = ns.substr(0, pos);
            obsVariants[orig].push_back(ns);
        }
    }

    std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> newRewards;

    for (const auto& sourceEntry : pomdp.rewards) {
        std::string origSource = sourceEntry.first;
        const auto& innerMap = sourceEntry.second;

        if (origSource.find(':') != std::string::npos) {
            newRewards[origSource] = innerMap;
            continue;
        }

        if (obsVariants.find(origSource) == obsVariants.end())
            continue;

        for (const auto& obsSource : obsVariants[origSource]) {
            auto& expandedRewardMap = newRewards[obsSource];

            for (const auto& rewardEntry : innerMap) {
                std::string innerKey = rewardEntry.first;
                size_t pos = innerKey.find(':');
                if (pos == std::string::npos)
                    continue;

                std::string actionPart = innerKey.substr(0, pos);
                std::string targetPart = innerKey.substr(pos + 1);

                if (targetPart.find(':') != std::string::npos) {
                    if (expandedRewardMap.find(innerKey) == expandedRewardMap.end()) {
                        expandedRewardMap[innerKey] = rewardEntry.second;
                        STORM_PRINT("Copied reward for key: " + innerKey);
                    }
                } else {
                    if (obsVariants.find(targetPart) != obsVariants.end()) {
                        for (const auto& obsTarget : obsVariants[targetPart]) {
                            std::string newInnerKey = actionPart + ":" + obsTarget;
                            expandedRewardMap[newInnerKey] = rewardEntry.second;
                            STORM_PRINT("Created reward for key: " + newInnerKey);
                        }
                    } else {
                        std::string newInnerKey = actionPart + ":" + targetPart;
                        expandedRewardMap[newInnerKey] = rewardEntry.second;
                        STORM_PRINT("Created reward for key (no obs variant): " + newInnerKey);
                    }
                }
            }
        }
    }

    pomdp.rewards = std::move(newRewards);
}

template<typename ValueType>
PomdpSolveParserResult<ValueType> PomdpSolveParser<ValueType>::parsePomdpSolveFile(const std::string& filename) {
    POMDPcomponents<ValueType> pomdp = parsePomdpFile<ValueType>(filename);

    createNewTransitions(pomdp);
    addInitialStateTransitions(pomdp);
    expandNewStatesTransitions(pomdp);
    filterDuplicateEntries(pomdp.newTransitions);

    std::vector<std::string> combinedStates;
    combinedStates.reserve(pomdp.states.size() + pomdp.newStates.size());
    combinedStates.insert(combinedStates.end(), pomdp.states.begin(), pomdp.states.end());
    combinedStates.insert(combinedStates.end(), pomdp.newStates.begin(), pomdp.newStates.end());

    auto transitionMatrix = buildTransitionMatrix<ValueType>(pomdp.newTransitions, combinedStates, pomdp.actions);

    std::unordered_map<std::string, uint64_t> stateIndices;
    uint64_t index = 0;
    for (const auto& state : combinedStates) {
        stateIndices[state] = index++;
    }

    storm::models::sparse::StateLabeling stateLabeling(combinedStates.size());
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", stateIndices["initial"]);

    expandObservationBasedRewards(pomdp);

    auto rewardMatrix = buildRewardMatrix<ValueType>(pomdp.newRewards, combinedStates, pomdp.actions);

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

    if (std::find(pomdp.observations.begin(), pomdp.observations.end(), "nothing") == pomdp.observations.end()) {
        pomdp.observations.push_back("nothing");
    }
    std::vector<uint32_t> stateObservations(combinedStates.size(), 0);
    for (uint64_t i = 0; i < combinedStates.size(); ++i) {
        const std::string& state = combinedStates[i];
        if (state == "initial") {
            auto it = std::find(pomdp.observations.begin(), pomdp.observations.end(), "start");
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
        } else {
            auto it = std::find(pomdp.observations.begin(), pomdp.observations.end(), "nothing");
            stateObservations[i] = static_cast<uint32_t>(std::distance(pomdp.observations.begin(), it));
        }
    }
    new_pomdp->updateObservations(std::move(stateObservations), true);

    std::string outputFilePath = "/home/kaloyank/storm/kaloyanFork/storm/resources/examples/testfiles/parser/example.pomdp.dot";
    auto modelPtr = new_pomdp->template as<storm::models::sparse::Model<ValueType>>();
    storm::api::exportSparseModelAsDot(modelPtr, outputFilePath);

    ValueType discountFactor = static_cast<ValueType>(pomdp.discount);

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
