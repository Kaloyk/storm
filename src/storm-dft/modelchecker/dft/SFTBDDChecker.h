#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "storm-dft/storage/SylvanBddManager.h"
#include "storm-dft/storage/dft/DFT.h"

namespace storm {
namespace modelchecker {

/**
 * Main class for the SFTBDDChecker
 *
 */
class SFTBDDChecker {
   public:
    using ValueType = double;
    using Bdd = sylvan::Bdd;

    SFTBDDChecker(std::shared_ptr<storm::storage::DFT<ValueType>> dft);

    SFTBDDChecker(
        std::shared_ptr<storm::storage::SylvanBddManager> sylvanBddManager,
        std::shared_ptr<storm::storage::DFT<ValueType>> dft);

    /**
     * Exports the Bdd that represents the top level gate to a file
     * in the dot format.
     *
     * \param filename
     * The name of the file the dot graph is written to
     */
    void exportBddToDot(std::string const &filename) {
        sylvanBddManager->exportBddToDot(getTopLevelGateBdd(), filename);
    }

    /**
     * \return
     * Generated Bdd that represents the formula of the top level gate
     */
    Bdd getTopLevelGateBdd();

    /**
     * \return
     * Generated Bdds that represent the logical formula of the given events
     */
    std::map<std::string, Bdd> getRelevantEventBdds(
        std::set<std::string> relevantEventNames);

    /**
     * \return
     * A set of minimal cut sets,
     * where the basic events are identified by their name
     */
    std::set<std::set<std::string>> getMinimalCutSets();

    /**
     * \return
     * The Probability that the top level gate fails.
     *
     * \note
     * Works only with exponential distributions and no spares.
     * Otherwise the function returns an arbitrary value
     */
    ValueType getProbabilityAtTimebound(ValueType timebound) {
        return getProbabilityAtTimebound(getTopLevelGateBdd(), timebound);
    }

    /**
     * \return
     * The Probabilities that the given Event fails at the given timebound.
     *
     * \param bdd
     * The bdd that represents an event in the dft.
     * Must be from a call to some function of *this.
     *
     * \note
     * Works only with exponential distributions and no spares.
     * Otherwise the function returns an arbitrary value
     */
    ValueType getProbabilityAtTimebound(Bdd bdd, ValueType timebound) const;

    /**
     * \return
     * The Probabilities that the top level gate fails at the given timepoints.
     *
     * \param timepoints
     * Array of timebounds to calculate the failure probabilities for.
     *
     * \param chunksize
     * Splits the timepoints array into chunksize chunks.
     * A value of 0 represents to calculate the whole array at once.
     *
     * \note
     * Works only with exponential distributions and no spares.
     * Otherwise the function returns an arbitrary value
     */
    std::vector<ValueType> getProbabilitiesAtTimepoints(
        std::vector<ValueType> const &timepoints, size_t const chunksize = 0) {
        return getProbabilitiesAtTimepoints(getTopLevelGateBdd(), timepoints,
                                            chunksize);
    }

    /**
     * \return
     * The Probabilities that the given Event fails at the given timepoints.
     *
     * \param bdd
     * The bdd that represents an event in the dft.
     * Must be from a call to some function of *this.
     *
     * \param timepoints
     * Array of timebounds to calculate the failure probabilities for.
     *
     * \param chunksize
     * Splits the timepoints array into chunksize chunks.
     * A value of 0 represents to calculate the whole array at once.
     *
     * \note
     * Works only with exponential distributions and no spares.
     * Otherwise the function returns an arbitrary value
     */
    std::vector<ValueType> getProbabilitiesAtTimepoints(
        Bdd bdd, std::vector<ValueType> const &timepoints,
        size_t chunksize = 0) const;

   private:
    std::map<uint64_t, std::map<uint64_t, Bdd>> withoutCache{};
    /**
     * The without operator as defined by Rauzy93
     * https://doi.org/10.1016/0951-8320(93)90060-C
     *
     * \node
     * f and g must be monotonic
     *
     * \return
     * f without paths that are included in a path in g
     */
    Bdd without(Bdd const f, Bdd const g);

    std::map<uint64_t, Bdd> minsolCache{};
    /**
     * The minsol algorithm as defined by Rauzy93
     * https://doi.org/10.1016/0951-8320(93)90060-C
     *
     * \node
     * f must be monotonic
     *
     * \return
     * A bdd encoding the minmal solutions of f
     */
    Bdd minsol(Bdd const f);

    /**
     * recursivly traverses the given BDD and returns the minimalCutSets
     *
     * \param bdd
     * The current bdd
     *
     * \param buffer
     * Reference to a vector that is used as a stack.
     * Temporarily stores the positive variables encountered.
     *
     * \param minimalCutSets
     * Reference to a set of minimal cut sets.
     * Will be populated by the function.
     */
    void recursiveMCS(Bdd const bdd, std::vector<uint32_t> &buffer,
                      std::set<std::set<std::string>> &minimalCutSets) const;

    std::shared_ptr<storm::storage::SylvanBddManager> sylvanBddManager;
    std::shared_ptr<storm::storage::DFT<ValueType>> dft;

    bool calculatedTopLevelGate;
    Bdd topLevelGateBdd;
};

}  // namespace modelchecker
}  // namespace storm
