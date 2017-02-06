#include "storm/settings/modules/IOSettings.h"

#include "storm/settings/SettingsManager.h"
#include "storm/settings/SettingMemento.h"
#include "storm/settings/Option.h"
#include "storm/settings/OptionBuilder.h"
#include "storm/settings/ArgumentBuilder.h"
#include "storm/settings/Argument.h"
#include "storm/exceptions/InvalidSettingsException.h"
#include "storm/parser/CSVParser.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/IllegalArgumentValueException.h"

namespace storm {
    namespace settings {
        namespace modules {
            
            const std::string IOSettings::moduleName = "io";
            const std::string IOSettings::exportDotOptionName = "exportdot";
            const std::string IOSettings::exportExplicitOptionName = "exportexplicit";
            const std::string IOSettings::explicitOptionName = "explicit";
            const std::string IOSettings::explicitOptionShortName = "exp";
            const std::string IOSettings::prismInputOptionName = "prism";
            const std::string IOSettings::janiInputOptionName = "jani";
            const std::string IOSettings::prismToJaniOptionName = "prism2jani";
            const std::string IOSettings::jitOptionName = "jit";
            const std::string IOSettings::explorationOrderOptionName = "explorder";
            const std::string IOSettings::explorationOrderOptionShortName = "eo";
            const std::string IOSettings::explorationChecksOptionName = "explchecks";
            const std::string IOSettings::explorationChecksOptionShortName = "ec";
            const std::string IOSettings::transitionRewardsOptionName = "transrew";
            const std::string IOSettings::stateRewardsOptionName = "staterew";
            const std::string IOSettings::choiceLabelingOptionName = "choicelab";
            const std::string IOSettings::constantsOptionName = "constants";
            const std::string IOSettings::constantsOptionShortName = "const";
            const std::string IOSettings::prismCompatibilityOptionName = "prismcompat";
            const std::string IOSettings::prismCompatibilityOptionShortName = "pc";
            const std::string IOSettings::noBuildOptionName = "nobuild";
            const std::string IOSettings::fullModelBuildOptionName = "buildfull";
            const std::string IOSettings::janiPropertyOptionName = "janiproperty";
            const std::string IOSettings::janiPropertyOptionShortName = "jprop";

            
            IOSettings::IOSettings() : ModuleSettings(moduleName) {
                this->addOption(storm::settings::OptionBuilder(moduleName, prismCompatibilityOptionName, false, "Enables PRISM compatibility. This may be necessary to process some PRISM models.").setShortName(prismCompatibilityOptionShortName).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, exportDotOptionName, "", "If given, the loaded model will be written to the specified file in the dot format.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file to which the model is to be written.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, exportExplicitOptionName, "", "If given, the loaded model will be written to the specified file in the drn format.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "the name of the file to which the model is to be writen.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, explicitOptionName, false, "Parses the model given in an explicit (sparse) representation.").setShortName(explicitOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("transition filename", "The name of the file from which to read the transitions.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build())
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("labeling filename", "The name of the file from which to read the state labeling.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, prismInputOptionName, false, "Parses the model given in the PRISM format.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file from which to read the PRISM input.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, janiInputOptionName, false, "Parses the model given in the JANI format.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file from which to read the JANI input.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, prismToJaniOptionName, false, "If set, the input PRISM model is transformed to JANI.").build());
                this->addOption(storm::settings::OptionBuilder(moduleName, jitOptionName, false, "If set, the model is built using the JIT model builder.").build());
                this->addOption(storm::settings::OptionBuilder(moduleName, fullModelBuildOptionName, false, "If set, include all rewards and labels.").build());
                this->addOption(storm::settings::OptionBuilder(moduleName, noBuildOptionName, false, "If set, do not build the model.").build());

                std::vector<std::string> explorationOrders = {"dfs", "bfs"};
                this->addOption(storm::settings::OptionBuilder(moduleName, explorationOrderOptionName, false, "Sets which exploration order to use.").setShortName(explorationOrderOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The name of the exploration order to choose.").addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(explorationOrders)).setDefaultValueString("bfs").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, explorationChecksOptionName, false, "If set, additional checks (if available) are performed during model exploration to debug the model.").setShortName(explorationChecksOptionShortName).build());

                this->addOption(storm::settings::OptionBuilder(moduleName, transitionRewardsOptionName, false, "If given, the transition rewards are read from this file and added to the explicit model. Note that this requires the model to be given as an explicit model (i.e., via --" + explicitOptionName + ").")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the transition rewards.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, stateRewardsOptionName, false, "If given, the state rewards are read from this file and added to the explicit model. Note that this requires the model to be given as an explicit model (i.e., via --" + explicitOptionName + ").")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the state rewards.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, choiceLabelingOptionName, false, "If given, the choice labels are read from this file and added to the explicit model. Note that this requires the model to be given as an explicit model (i.e., via --" + explicitOptionName + ").")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the choice labels.").addValidatorString(ArgumentValidatorFactory::createExistingFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, constantsOptionName, false, "Specifies the constant replacements to use in symbolic models. Note that this requires the model to be given as an symbolic model (i.e., via --" + prismInputOptionName + " or --" + janiInputOptionName + ").").setShortName(constantsOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("values", "A comma separated list of constants and their value, e.g. a=1,b=2,c=3.").setDefaultValueString("").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, janiPropertyOptionName, false, "Specifies the properties from the jani model (given by --" + janiInputOptionName + ")  to be checked.").setShortName(janiPropertyOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("values", "A comma separated list of properties to be checked").setDefaultValueString("").build()).build());
            }

            bool IOSettings::isExportDotSet() const {
                return this->getOption(exportDotOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getExportDotFilename() const {
                return this->getOption(exportDotOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool IOSettings::isExportExplicitSet() const {
                return this->getOption(exportExplicitOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getExportExplicitFilename() const {
                return this->getOption(exportExplicitOptionName).getArgumentByName("filename").getValueAsString();
            }

            bool IOSettings::isExplicitSet() const {
                return this->getOption(explicitOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getTransitionFilename() const {
                return this->getOption(explicitOptionName).getArgumentByName("transition filename").getValueAsString();
            }
                        
            std::string IOSettings::getLabelingFilename() const {
                return this->getOption(explicitOptionName).getArgumentByName("labeling filename").getValueAsString();
            }
            
            bool IOSettings::isPrismInputSet() const {
                return this->getOption(prismInputOptionName).getHasOptionBeenSet();
            }
            
            bool IOSettings::isPrismOrJaniInputSet() const {
                return isJaniInputSet() || isPrismInputSet();
            }
            
            bool IOSettings::isPrismToJaniSet() const {
                return this->getOption(prismToJaniOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getPrismInputFilename() const {
                return this->getOption(prismInputOptionName).getArgumentByName("filename").getValueAsString();
            }

            bool IOSettings::isJaniInputSet() const {
                return this->getOption(janiInputOptionName).getHasOptionBeenSet();
            }

            std::string IOSettings::getJaniInputFilename() const {
                return this->getOption(janiInputOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool IOSettings::isJitSet() const {
                return this->getOption(jitOptionName).getHasOptionBeenSet();
            }

            bool IOSettings::isExplorationOrderSet() const {
                return this->getOption(explorationOrderOptionName).getHasOptionBeenSet();
            }
            
            storm::builder::ExplorationOrder IOSettings::getExplorationOrder() const {
                std::string explorationOrderAsString = this->getOption(explorationOrderOptionName).getArgumentByName("name").getValueAsString();
                if (explorationOrderAsString == "dfs") {
                    return storm::builder::ExplorationOrder::Dfs;
                } else if (explorationOrderAsString == "bfs") {
                    return storm::builder::ExplorationOrder::Bfs;
                }
                STORM_LOG_THROW(false, storm::exceptions::IllegalArgumentValueException, "Unknown exploration order '" << explorationOrderAsString << "'.");
            }
            
            bool IOSettings::isExplorationChecksSet() const {
                return this->getOption(explorationChecksOptionName).getHasOptionBeenSet();
            }
            
            bool IOSettings::isTransitionRewardsSet() const {
                return this->getOption(transitionRewardsOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getTransitionRewardsFilename() const {
                return this->getOption(transitionRewardsOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool IOSettings::isStateRewardsSet() const {
                return this->getOption(stateRewardsOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getStateRewardsFilename() const {
                return this->getOption(stateRewardsOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool IOSettings::isChoiceLabelingSet() const {
                return this->getOption(choiceLabelingOptionName).getHasOptionBeenSet();
            }
                
            std::string IOSettings::getChoiceLabelingFilename() const {
                return this->getOption(choiceLabelingOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            std::unique_ptr<storm::settings::SettingMemento> IOSettings::overridePrismCompatibilityMode(bool stateToSet) {
                return this->overrideOption(prismCompatibilityOptionName, stateToSet);
            }
            
            bool IOSettings::isConstantsSet() const {
                return this->getOption(constantsOptionName).getHasOptionBeenSet();
            }
            
            std::string IOSettings::getConstantDefinitionString() const {
                return this->getOption(constantsOptionName).getArgumentByName("values").getValueAsString();
            }
            
            bool IOSettings::isJaniPropertiesSet() const {
                return this->getOption(janiPropertyOptionName).getHasOptionBeenSet();
            }
            
            std::vector<std::string> IOSettings::getJaniProperties() const {
                return storm::parser::parseCommaSeperatedValues(this->getOption(janiPropertyOptionName).getArgumentByName("values").getValueAsString());
            }

            bool IOSettings::isPrismCompatibilityEnabled() const {
                return this->getOption(prismCompatibilityOptionName).getHasOptionBeenSet();
            }
            
            bool IOSettings::isBuildFullModelSet() const {
                return this->getOption(fullModelBuildOptionName).getHasOptionBeenSet();
            }
            
            bool IOSettings::isNoBuildModelSet() const {
                return this->getOption(noBuildOptionName).getHasOptionBeenSet();
            }

			void IOSettings::finalize() {
            }

            bool IOSettings::check() const {
                // Ensure that not two symbolic input models were given.
                STORM_LOG_THROW(!isJaniInputSet() || !isPrismInputSet(), storm::exceptions::InvalidSettingsException, "Symbolic model ");
                
                // Ensure that the model was given either symbolically or explicitly.
                STORM_LOG_THROW(!isJaniInputSet() || !isPrismInputSet() || !isExplicitSet(), storm::exceptions::InvalidSettingsException, "The model may be either given in an explicit or a symbolic format (PRISM or JANI), but not both.");
                
                // Make sure PRISM-to-JANI conversion is only set if the actual input is in PRISM format.
                STORM_LOG_THROW(!isPrismToJaniSet() || isPrismInputSet(), storm::exceptions::InvalidSettingsException, "For the transformation from PRISM to JANI, the input model must be given in the prism format.");
                
                return true;
            }

        } // namespace modules
    } // namespace settings
} // namespace storm
