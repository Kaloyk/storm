#include "PomdpSolveParser.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/api/builder.h"
#include "Utility.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <iostream>

namespace storm {
namespace pomdp {
namespace parser {

struct POMDP {
    std::vector<std::string> states;
    std::vector<std::string> actions;
    std::vector<std::string> observations;
    std::vector<double> start_probabilities;
    std::map<std::string, std::map<std::string, double>> transitions;
    std::map<std::string, std::map<std::string, double>> observations_prob;
    std::map<std::string, double> rewards;
    std::vector<std::string> newStates; // Stores unique new states with observation incorporated
    std::map<std::string, std::map<std::string, double>> newTransitions; // Modified transitions with observation-based states

    std::map<std::string, std::map<std::string, double>> createNewTransitions() {
        std::unordered_set<std::string> uniqueStates;

        for (const auto& action_entry : transitions) {
            const std::string& action = action_entry.first;

            for (const auto& transition_entry : action_entry.second) {
                auto pos = transition_entry.first.find(':');
                std::string startState = transition_entry.first.substr(0, pos);
                std::string endState = transition_entry.first.substr(pos + 1);
                double transition_prob = transition_entry.second;

                double totalProbability = 0.0;

                if (observations_prob.count(action) > 0) {
                    std::string obs_key_prefix = endState + ":";
                    for (const auto& obs_entry : observations_prob.at(action)) {
                        if (obs_entry.first.rfind(obs_key_prefix, 0) == 0) {
                            std::string observation = obs_entry.first.substr(obs_key_prefix.length());
                            double obs_prob = obs_entry.second;

                            double new_prob = transition_prob * obs_prob;
                            std::string newEndState = endState + ":" + observation;

                            if (new_prob > 0) {
                                std::string actionEndStateKey = action + ":" + newEndState;
                                newTransitions[startState][actionEndStateKey] = new_prob;
                                totalProbability += new_prob;
                                uniqueStates.insert(newEndState);
                            }
                        }
                    }
                }

                if (totalProbability < transition_prob) {
                    double remaining_prob = transition_prob - totalProbability;
                    std::string actionEndStateKey = action + ":" + endState;
                    newTransitions[startState][actionEndStateKey] = remaining_prob;
                }
            }
        }

        newStates.assign(uniqueStates.begin(), uniqueStates.end());
        return newTransitions;
    }

    void addInitialStateTransitions() {
        const std::string initialState = "initial";
        const std::string action = "start";

        if (start_probabilities.size() != states.size()) {
            std::cerr << "Error: start_probabilities and states vectors must be of the same size." << std::endl;
            return;
        }

        for (size_t i = 0; i < start_probabilities.size(); ++i) {
            double probability = start_probabilities[i];
            const std::string& state = states[i];

            if (probability > 0.0) {
                std::string newState = state + ":start";
                std::string actionEndStateKey = action + ":" + newState;
                newTransitions[initialState][actionEndStateKey] = probability;
                newStates.push_back(newState);
            }
        }
    }

    void expandNewStatesTransitions() {
        std::map<std::string, std::map<std::string, double>> additionalTransitions;

        for (const auto& newState : newStates) {
            auto pos = newState.find(':');
            if (pos == std::string::npos) continue;

            std::string originalState = newState.substr(0, pos);

            for (const auto& state_entry : newTransitions) {
                const std::string& startState = state_entry.first;

                if (startState == originalState) {
                    for (const auto& actionEntry : state_entry.second) {
                        std::string newActionKey = actionEntry.first;
                        additionalTransitions[newState][newActionKey] = actionEntry.second;
                    }
                }
            }
        }

        for (const auto& state_entry : additionalTransitions) {
            for (const auto& actionEntry : state_entry.second) {
                newTransitions[state_entry.first][actionEntry.first] = actionEntry.second;
            }
        }
    }
};

POMDP parsePOMDPFile(const std::string &filename) {
    POMDP pomdp;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        if (word == "states:") {
            while (iss >> word) {
                pomdp.states.push_back(trim(word));
            }
        } else if (word == "actions:") {
            while (iss >> word) {
                pomdp.actions.push_back(trim(word));
            }
        } else if (word == "observations:") {
            while (iss >> word) {
                pomdp.observations.push_back(trim(word));
            }
        } else if (word == "start:") {
            double prob;
            while (iss >> prob) {
                pomdp.start_probabilities.push_back(prob); // Store each probability
            }
        } else if (word == "T:") {
            std::string action, start_state, end_state;
            std::vector<double> probabilities;

            std::getline(iss, word, ':');  // Skip 'T:' part
            if (iss.peek() == ':') {
                std::getline(iss, action, ':');  // Action
                action = trim(action); 
                if (iss.peek() == ':') {
                    // Single Transition Probability (T: <action> : <start-state> : <end-state> %f)
                    std::getline(iss, start_state, ':');
                    std::getline(iss, end_state);
                    double probability;
                    iss >> probability;

                    start_state = trim(start_state);
                    end_state = trim(end_state);

                    if (probability != 0) {
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
                } else {
                    // Single Row of Transition Matrix (T: <action> : <start-state> %f %f ... %f)
                    std::getline(iss, action, ':');
                    action = trim(action);
                    iss >> start_state;
                    start_state = trim(start_state);
                    double prob;
                    while (iss >> prob) {
                        probabilities.push_back(prob);
                    }

                    for (int i = 0; i < pomdp.states.size(); i++) {
                        if (probabilities[i] != 0) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.transitions[act][start_state + ":" + pomdp.states[i]] = probabilities[i];
                                }
                            } else {
                                pomdp.transitions[action][start_state + ":" + pomdp.states[i]] = probabilities[i];
                            }
                        }
                    }
                }
            } else {
                // Entire Transition Matrix for an Action (T: <action>)
                action = trim(line.substr(3)); // Get action from "T: <action>"

                for (int i = 0; i < pomdp.states.size(); i++) {
                    std::getline(infile, line); // Read each row
                    std::istringstream rowStream(line);
                    for (int j = 0; j < pomdp.states.size(); j++) {
                        double prob;
                        rowStream >> prob;
                        if (prob != 0) {
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
            std::string action, end_state, observation;
            std::vector<double> probabilities;

            std::getline(iss, word, ':'); // Skip 'O:' part
            if (iss.peek() == ':') {
                std::getline(iss, action, ':');
                action = trim(action);
                if (iss.peek() == ':') {
                    // Single Observation Probability (O: <action> : <end-state> : <observation> %f)
                    std::getline(iss, end_state, ':');
                    std::getline(iss, observation);
                    double probability;
                    iss >> probability;

                    end_state = trim(end_state);
                    observation = trim(observation);

                    if (probability != 0) {
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
                } else {
                    // Single Row of Observation Matrix (O: <action> : <end-state> %f %f ... %f)
                    std::getline(iss, end_state);
                    end_state = trim(end_state);
                    double prob;
                    while (iss >> prob) {
                        probabilities.push_back(prob);
                    }

                    for (int i = 0; i < pomdp.observations.size(); i++) {
                        if (probabilities[i] != 0) {
                            if (action == "*") {
                                for (const auto& act : pomdp.actions) {
                                    pomdp.observations_prob[act][end_state + ":" + pomdp.observations[i]] = probabilities[i];
                                }
                            } else {
                                pomdp.observations_prob[action][end_state + ":" + pomdp.observations[i]] = probabilities[i];
                            }
                        }
                    }
                }
            } else {
                // Entire Observation Matrix for an Action (O: <action>)
                action = trim(line.substr(3)); // Get action from "O: <action>"

                for (int i = 0; i < pomdp.states.size(); i++) {
                    std::getline(infile, line); // Read each row
                    std::istringstream rowStream(line);
                    for (int j = 0; j < pomdp.observations.size(); j++) {
                        double prob;
                        rowStream >> prob;
                        if (prob != 0) {
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
            std::string action, from_state, to_state, observation;
            double reward;
            iss >> action >> from_state >> to_state >> observation >> reward;
            pomdp.rewards[trim(from_state) + ":" + trim(action) + ":" + trim(to_state) + ":" + trim(observation)] = reward;
        }
    }

    return pomdp;
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> buildTransitionMatrix(
    const std::map<std::string, std::map<std::string, double>>& newTransitions,
    const std::vector<std::string>& newStates,
    const std::vector<std::string>& actions
) {
    // Anzahl der States und Choices
    uint64_t numStates = newStates.size();
    uint64_t numChoices = newTransitions.size() * actions.size();
    
    // Map zum schnellen Lookup von State-Indices
    std::map<std::string, uint64_t> stateIndices;
    for (uint64_t i = 0; i < numStates; ++i) {
        stateIndices[newStates[i]] = i;
    }

    // Zähle die Anzahl der Nicht-Null-Einträge (entries)
    uint64_t entryCount = 0;
    for (const auto& [startState, actionMap] : newTransitions) {
        entryCount += actionMap.size();
    }

    // SparseMatrixBuilder initialisieren
    storm::storage::SparseMatrixBuilder<ValueType> builder(
        numChoices,    // Anzahl der Rows (choices)
        numStates,     // Anzahl der Columns (states)
        entryCount,    // Anzahl der Nicht-Null-Einträge
        true,          // forceDimensions
        true,          // hasCustomRowGrouping
        numStates      // Anzahl der RowGroups (states)
    );

    // RowGroup-Logik für States
    uint64_t rowIndex = 0; // Laufender Index für Rows
    for (const auto& [startState, actionMap] : newTransitions) {
        if (stateIndices.find(startState) == stateIndices.end()) {
            throw std::runtime_error("StartState not found in newStates: " + startState);
        }

        // Markiere den Beginn der RowGroup
        uint64_t rowGroupStart = rowIndex;
        builder.newRowGroup(rowGroupStart);

        // Gehe durch alle Aktionen und Ziele
        for (const auto& [actionEndState, probability] : actionMap) {
            auto pos = actionEndState.find(':');
            if (pos == std::string::npos) {
                throw std::runtime_error("Invalid action:endState format: " + actionEndState);
            }

            // Extrahiere EndState
            std::string endState = actionEndState.substr(pos + 1);
            if (stateIndices.find(endState) == stateIndices.end()) {
                throw std::runtime_error("EndState not found in newStates: " + endState);
            }

            uint64_t colIndex = stateIndices[endState]; // Zielzustand als Spalte
            builder.addNextValue(rowIndex, colIndex, probability); // Füge Eintrag hinzu
        }

        // Nächste Row
        ++rowIndex;
    }

    // TransitionMatrix erstellen
    return builder.build();
}


template<typename ValueType>
PomdpSolveParserResult<ValueType> PomdpSolveParser<ValueType>::parsePomdpSolveFile(std::string const& filename) {
    STORM_LOG_WARN("POMDPsolve parser not implemented yet.");
    POMDP pomdp = parsePOMDPFile(filename);
    pomdp.createNewTransitions();
    pomdp.addInitialStateTransitions();
    pomdp.expandNewStatesTransitions();
    auto transitionMatrix = buildTransitionMatrix<ValueType>(
            pomdp.newTransitions,
            pomdp.newStates,
            pomdp.actions
        );

    // Create a placeholder StateLabeling
    storm::models::sparse::StateLabeling stateLabeling; // Empty state labeling

    // Create a placeholder reward model map
    std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> rewardModels;

    // Construct the sparse POMDP
    auto new_pomdp = std::make_shared<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>(
        transitionMatrix,
        stateLabeling,
        rewardModels
    );

    ValueType discountFactor;
    // Prepare the result struct
    PomdpSolveParserResult<ValueType> result;
    result.pomdp = new_pomdp;
    result.discountFactor = discountFactor;

    return result;
}

template class PomdpSolveParser<double>;
template class PomdpSolveParser<storm::RationalNumber>;

}  // namespace parser
}  // namespace pomdp
}  // namespace storm
