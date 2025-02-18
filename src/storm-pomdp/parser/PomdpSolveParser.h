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