#include "storm-pomdp/modelchecker/BeliefExplorationPomdpModelChecker.h"
#include "storm/environment/Environment.h"

namespace storm {
namespace pomdp {
namespace api {

/**
 * Uses the belief exploration with cut-offs to under-approximate the given objective on a POMDP.
 * @tparam ValueType number type to be used
 * @param env the environment to use
 * @param pomdp the input pomdp to be checked
 * @param task the check task to be performed
 * @param sizeThreshold number of states up to which the belief MDP should be unfolded
 * @param pomdpStateValues additional values that can be used for cut-offs in the under-approximation (generated by finite memory schedulers).
 * Each element of the outer vector represents a scheduler. Each scheduler itself is represented by a vector of maps representing (memory node x state) -> value
 * @return the result structure
 */
template<typename ValueType>
typename storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
    storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>::Result
underapproximateWithCutoffs(storm::Environment const& env,
                            std::shared_ptr<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>> pomdp,
                            storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& task, uint64_t sizeThreshold,
                            std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> additionalPomdpStateValues =
                                std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>()) {
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelCheckerOptions<ValueType> options(false, true);
    options.useClipping = false;
    options.useStateEliminationCutoff = false;
    options.sizeThresholdInit = sizeThreshold;
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>
        modelchecker(pomdp, options);
    return modelchecker.check(task.getFormula(), additionalPomdpStateValues);
}

/**
 * Uses the belief exploration with cut-offs *without* the pre-processing to generate cut-off values to under-approximate the given objective on a POMDP.
 * Cut-off values need to be provided in the form of a vector of vectors representing finite memory schedulers.
 * @tparam ValueType number type to be used
 * @param env the environment to use
 * @param pomdp the input pomdp to be checked
 * @param task the check task to be performed
 * @param sizeThreshold number of states up to which the belief MDP should be unfolded
 * @param pomdpStateValues additional values that can be used for cut-offs in the under-approximation (generated by finite memory schedulers).
 * Each element of the outer vector represents a scheduler. Each scheduler itself is represented by a vector of maps representing (memory node x state) -> value
 * @return the result structure
 */
template<typename ValueType>
typename storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
    storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>::Result
underapproximateWithoutHeuristicValues(storm::Environment const& env,
                                       std::shared_ptr<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>> pomdp,
                                       storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& task, uint64_t sizeThreshold,
                                       std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> pomdpStateValues) {
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelCheckerOptions<ValueType> options(false, true);
    options.skipHeuristicSchedulers = true;
    options.useClipping = false;
    options.useStateEliminationCutoff = false;
    options.sizeThresholdInit = sizeThreshold;
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>
        modelchecker(pomdp, options);
    return modelchecker.check(task.getFormula(), pomdpStateValues);
}

// Interactive Interface
/**
 * Create a model checker with the correct settings for an interactive unfolding. Needs to be called first to set-up the unfolding.
 * @tparam ValueType number type to be used
 * @param env the environment to use
 * @param pomdp the input pomdp to be checked
 * @param useClipping true if clipping is to be used in addition to cut-offs
 * @return the model checker object, configured for an interactive unfolding
 */
template<typename ValueType>
storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>
createInteractiveUnfoldingModelChecker(storm::Environment const& env,
                                       std::shared_ptr<storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>> pomdp,
                                       bool useClipping) {
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelCheckerOptions<ValueType> options(false, true);
    options.skipHeuristicSchedulers = false;
    options.useClipping = useClipping;
    options.useStateEliminationCutoff = false;
    options.sizeThresholdInit = storm::utility::infinity<uint64_t>();
    options.interactiveUnfolding = true;
    options.refine = false;
    options.gapThresholdInit = 0;
    options.cutZeroGap = false;
    storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>
        modelchecker(pomdp, options);
    return modelchecker;
}

/**
 * Start an interactive unfolding to under approximate the given objective
 * @tparam ValueType number type to be used
 * @param modelchecker the model checker object configured for the interactive unfolding
 * @param task the check task to be performed
 * @param additionalPomdpStateValues additional values that can be used for cut-offs in the under-approximation (generated by finite memory schedulers).
 * Each element of the outer vector represents a scheduler. Each scheduler itself is represented by a vector of maps representing (memory node x state) -> value
 */
template<typename ValueType>
void startInteractiveExploration(storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
                                     storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>& modelchecker,
                                 storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& task,
                                 std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>> additionalPomdpStateValues =
                                     std::vector<std::vector<std::unordered_map<uint64_t, ValueType>>>()) {
    modelchecker.check(task.getFormula(), additionalPomdpStateValues);
}

/**
 * Extract the scheduler generated by an under-approximation from the given result struct. The scheduler is represented by a Markov chain.
 * @tparam ValueType number type to be used
 * @param modelcheckingResult the result struct containing the scheduler
 * @return the scheduler represented by a Markov chain.
 */
template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<ValueType>> extractSchedulerAsMarkovChain(
    typename storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>::Result modelcheckingResult) {
    return modelcheckingResult.schedulerAsMarkovChain;
}

/**
 * Get a specific scheduler used to generate cut-off values from the result struct.
 * @tparam ValueType  number type to be used
 * @param modelcheckingResult the result struct
 * @param schedId the ID of the scheduler used during the exploration. This corresponds to the labels in the scheduler MC.
 * @return the desired scheduler
 */
template<typename ValueType>
storm::storage::Scheduler<ValueType> getCutoffScheduler(
    typename storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>::Result modelcheckingResult,
    uint64_t schedId) {
    return modelcheckingResult.cutoffSchedulers.at(schedId);
}

/**
 * Get the overall number of schedulers generated by the pre-processing for the under-approximation from the result struct.
 * @tparam ValueType number type to be used
 * @param modelcheckingResult the result struct
 * @return the number of schedulers
 */
template<typename ValueType>
uint64_t getNumberOfPreprocessingSchedulers(
    typename storm::pomdp::modelchecker::BeliefExplorationPomdpModelChecker<
        storm::models::sparse::Pomdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>>::Result modelcheckingResult) {
    return modelcheckingResult.cutoffSchedulers.size();
}
}  // namespace api
}  // namespace pomdp
}  // namespace storm