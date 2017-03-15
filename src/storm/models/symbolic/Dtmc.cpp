#include "storm/models/symbolic/Dtmc.h"

#include "storm/storage/dd/DdManager.h"
#include "storm/storage/dd/Add.h"
#include "storm/storage/dd/Bdd.h"

#include "storm/models/symbolic/StandardRewardModel.h"

#include "storm/adapters/CarlAdapter.h"

namespace storm {
    namespace models {
        namespace symbolic {
            
            template<storm::dd::DdType Type, typename ValueType>
            Dtmc<Type, ValueType>::Dtmc(std::shared_ptr<storm::dd::DdManager<Type>> manager,
                                        storm::dd::Bdd<Type> reachableStates,
                                        storm::dd::Bdd<Type> initialStates,
                                        storm::dd::Bdd<Type> deadlockStates,
                                        storm::dd::Add<Type, ValueType> transitionMatrix,
                                        std::set<storm::expressions::Variable> const& rowVariables,
                                        std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter,
                                        std::set<storm::expressions::Variable> const& columnVariables,
                                        std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter,
                                        std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                                        std::map<std::string, storm::expressions::Expression> labelToExpressionMap,
                                        std::unordered_map<std::string, RewardModelType> const& rewardModels)
            : DeterministicModel<Type, ValueType>(storm::models::ModelType::Dtmc, manager, reachableStates, initialStates, deadlockStates, transitionMatrix, rowVariables, rowExpressionAdapter, columnVariables, columnExpressionAdapter, rowColumnMetaVariablePairs, labelToExpressionMap, rewardModels) {
                // Intentionally left empty.
            }
            
            // Explicitly instantiate the template class.
            template class Dtmc<storm::dd::DdType::CUDD, double>;
            template class Dtmc<storm::dd::DdType::Sylvan, double>;

            template class Dtmc<storm::dd::DdType::Sylvan, storm::RationalFunction>;

        } // namespace symbolic
    } // namespace models
} // namespace storm
