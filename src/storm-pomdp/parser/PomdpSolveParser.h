#pragma once
#include <string>

namespace storm {
namespace models {
namespace sparse {
// Forward declaration
template<typename ValueType, typename RewardModelType>
class Pomdp;
template<typename ValueType>
class StandardRewardModel;
}  // namespace sparse
}  // namespace models
namespace pomdp {
namespace parser {

struct PomdpData {
    double discountFactor;
    std::vector<std::string> states;
    std::vector<std::string> actions;
    std::vector<std::string> observations;
    std::unordered_map<std::string, double> transitionProbabilities;
    std::unordered_map<std::string, double> observationProbabilities;
    std::unordered_map<std::string, double> rewards;
};

template<typename ValueType>
struct IntermediatePomdpData {
    std::unordered_map<std::tuple<int, int, int>, ValueType> transitions; // [state, action, next_state] -> probability
    std::unordered_map<std::tuple<int, int, int>, ValueType> rewards; // [state, action, next_state] -> reward
    std::vector<ValueType> startStateDistribution; // Initial state distribution
};

// Declaration of the global variable to store POMDP data.
extern PomdpData globalPomdpData;

// Function to parse the POMDP file and populate the global data structure.
void parsePomdpFile(const std::string& filename);

template<typename ValueType>
struct PomdpSolveParserResult {
    std::shared_ptr<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>> pomdp;
    ValueType discountFactor;
    // Extend with what needs to be returned from the file.
};

template<typename ValueType>
class PomdpSolveParser {
   public:
    /*!
     * Parse POMDP in POMDP solve format and build POMDP.
     *
     * @param filename File.
     *
     * @return what needs to be returned from the file.
     */
    static PomdpSolveParserResult<ValueType> parsePomdpSolveFile(std::string const& filename);

    // Add more functions as needed.

   private:
    // Add more functions and members as needed.
};
}  // namespace parser
}  // namespace pomdp
}  // namespace storm