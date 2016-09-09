#ifndef STORM_SETTINGS_MODULES_IOSETTINGS_H_
#define STORM_SETTINGS_MODULES_IOSETTINGS_H_

#include "storm-config.h"
#include "src/settings/modules/ModuleSettings.h"

#include "src/builder/ExplorationOrder.h"

namespace storm {
    namespace settings {
        namespace modules {

            /*!
             * This class represents the markov chain settings.
             */
            class IOSettings : public ModuleSettings {
            public:

                /*!
                 * Creates a new set of IO settings.
                 */
                IOSettings();

                /*!
                 * Retrieves whether the export-to-dot option was set.
                 *
                 * @return True if the export-to-dot option was set.
                 */
                bool isExportDotSet() const;

                /*!
                 * Retrieves the name in which to write the model in dot format, if the export-to-dot option was set.
                 *
                 * @return The name of the file in which to write the exported model.
                 */
                std::string getExportDotFilename() const;

                /*!
                 * Retrieves whether the explicit option was set.
                 *
                 * @return True if the explicit option was set.
                 */
                bool isExplicitSet() const;

                /*!
                 * Retrieves the name of the file that contains the transitions if the model was given using the explicit
                 * option.
                 *
                 * @return The name of the file that contains the transitions.
                 */
                std::string getTransitionFilename() const;

                /*!
                 * Retrieves the name of the file that contains the state labeling if the model was given using the
                 * explicit option.
                 *
                 * @return The name of the file that contains the state labeling.
                 */
                std::string getLabelingFilename() const;

                /*!
                 * Retrieves whether the PRISM language option was set.
                 *
                 * @return True if the PRISM input option was set.
                 */
                bool isPrismInputSet() const;
                
                /*!
                 * Retrieves whether the JANI input option was set.
                 *
                 * @return True if the JANI input option was set.
                 */
                bool isJaniInputSet() const;

                /*!
                 * Retrieves whether the JANI or PRISM input option was set.
                 *
                 * @return True if either of the two options was set.
                 */
                bool isPrismOrJaniInputSet() const;
                
                /*!
                 * Retrieves the name of the file that contains the PRISM model specification if the model was given
                 * using the PRISM input option.
                 *
                 * @return The name of the file that contains the PRISM model specification.
                 */
                std::string getPrismInputFilename() const;

                /*!
                 * Retrieves the name of the file that contains the JANI model specification if the model was given
                 * using the JANI input option.
                 *
                 * @return The name of the file that contains the JANI model specification.
                 */
                std::string getJaniInputFilename() const;

                /*!
                 * Retrieves whether the model exploration order was set.
                 *
                 * @return True if the model exploration option was set.
                 */
                bool isExplorationOrderSet() const;
                
                /*!
                 * Retrieves the exploration order if it was set.
                 *
                 * @return The chosen exploration order.
                 */
                storm::builder::ExplorationOrder getExplorationOrder() const;

                /*!
                 * Retrieves whether the transition reward option was set.
                 *
                 * @return True if the transition reward option was set.
                 */
                bool isTransitionRewardsSet() const;

                /*!
                 * Retrieves the name of the file that contains the transition rewards if the model was given using the
                 * explicit option.
                 *
                 * @return The name of the file that contains the transition rewards.
                 */
                std::string getTransitionRewardsFilename() const;

                /*!
                 * Retrieves whether the state reward option was set.
                 *
                 * @return True if the state reward option was set.
                 */
                bool isStateRewardsSet() const;

                /*!
                 * Retrieves the name of the file that contains the state rewards if the model was given using the
                 * explicit option.
                 *
                 * @return The name of the file that contains the state rewards.
                 */
                std::string getStateRewardsFilename() const;

                /*!
                 * Retrieves whether the choice labeling option was set.
                 * 
                 * @return True iff the choice labeling option was set.
                 */
                bool isChoiceLabelingSet() const;

                /*!
                 * Retrieves the name of the file that contains the choice labeling
                 * if the model was given using the explicit option.
                 *
                 * @return The name of the file that contains the choice labeling.
                 */
                std::string getChoiceLabelingFilename() const;

                /*!
                 * Overrides the option to enable the PRISM compatibility mode by setting it to the specified value. As
                 * soon as the returned memento goes out of scope, the original value is restored.
                 *
                 * @param stateToSet The value that is to be set for the option.
                 * @return The memento that will eventually restore the original value.
                 */
                std::unique_ptr<storm::settings::SettingMemento> overridePrismCompatibilityMode(bool stateToSet);
                
                /*!
                 * Retrieves whether the export-to-dot option was set.
                 *
                 * @return True if the export-to-dot option was set.
                 */
                bool isConstantsSet() const;

                /*!
                 * Retrieves the string that defines the constants of a symbolic model (given via the symbolic option).
                 *
                 * @return The string that defines the constants of a symbolic model.
                 */
                std::string getConstantDefinitionString() const;

                /*!
                 * Retrieves whether the PRISM compatibility mode was enabled.
                 *
                 * @return True iff the PRISM compatibility mode was enabled.
                 */
                bool isPrismCompatibilityEnabled() const;

                bool check() const override;
                void finalize() override;

                // The name of the module.
                static const std::string moduleName;

            private:
                // Define the string names of the options as constants.
                static const std::string exportDotOptionName;
                static const std::string exportMatOptionName;
                static const std::string explicitOptionName;
                static const std::string explicitOptionShortName;
                static const std::string prismInputOptionName;
                static const std::string janiInputOptionName;
                static const std::string explorationOrderOptionName;
                static const std::string explorationOrderOptionShortName;
                static const std::string transitionRewardsOptionName;
                static const std::string stateRewardsOptionName;
                static const std::string choiceLabelingOptionName;
                static const std::string constantsOptionName;
                static const std::string constantsOptionShortName;
                static const std::string prismCompatibilityOptionName;
                static const std::string prismCompatibilityOptionShortName;
            };

        } // namespace modules
    } // namespace settings
} // namespace storm

#endif /* STORM_SETTINGS_MODULES_IOSETTINGS_H_ */
