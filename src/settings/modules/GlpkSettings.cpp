#include "src/settings/modules/GlpkSettings.h"

#include "src/settings/SettingsManager.h"

namespace storm {
    namespace settings {
        namespace modules {
            
            const std::string GlpkSettings::moduleName = "glpk";
            const std::string GlpkSettings::integerToleranceOption = "inttol";
            const std::string GlpkSettings::outputOptionName = "output";
            
            GlpkSettings::GlpkSettings(storm::settings::SettingsManager& settingsManager) : ModuleSettings(settingsManager, moduleName) {
                this->addAndRegisterOption(storm::settings::OptionBuilder(moduleName, outputOptionName, true, "If set, the glpk output will be printed to the command line.").build());
                this->addAndRegisterOption(storm::settings::OptionBuilder(moduleName, integerToleranceOption, true, "Sets glpk's precision for integer variables.").addArgument(storm::settings::ArgumentBuilder::createDoubleArgument("value", "The precision to achieve.").setDefaultValueDouble(1e-06).addValidationFunctionDouble(storm::settings::ArgumentValidators::doubleRangeValidatorExcluding(0.0, 1.0)).build()).build());
            }
            
        } // namespace modules
    } // namespace settings
} // namespace storm