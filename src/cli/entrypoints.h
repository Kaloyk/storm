#ifndef STORM_ENTRYPOINTS_H_H
#define STORM_ENTRYPOINTS_H_H

#include "src/utility/storm.h"

#include "src/exceptions/NotImplementedException.h"

namespace storm {
    namespace cli {

        template<typename ValueType>
        void verifySparseModel(std::shared_ptr<storm::models::sparse::Model<ValueType>> model, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            for (auto const& formula : formulas) {
                std::cout << std::endl << "Model checking property: " << *formula << " ...";
                std::unique_ptr<storm::modelchecker::CheckResult> result(storm::verifySparseModel(model, formula));
                if (result) {
                    std::cout << " done." << std::endl;
                    std::cout << "Result (initial states): ";
                    result->filter(storm::modelchecker::ExplicitQualitativeCheckResult(model->getInitialStates()));
                    std::cout << *result << std::endl;
                } else {
                    std::cout << " skipped, because the modelling formalism is currently unsupported." << std::endl;
                }
            }
        }

#ifdef STORM_HAVE_CARL
        template<>
        inline void verifySparseModel(std::shared_ptr<storm::models::sparse::Model<storm::RationalFunction>> model, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {

            for (auto const& formula : formulas) {
                STORM_LOG_THROW(model->getType() == storm::models::ModelType::Dtmc, storm::exceptions::InvalidSettingsException, "Currently parametric verification is only available for DTMCs.");
                std::cout << std::endl << "Model checking property: " << *formula << " ...";
                std::unique_ptr<storm::modelchecker::CheckResult> result(storm::verifySparseModel(model, formula));
                if (result) {
                    std::cout << " done." << std::endl;
                    std::cout << "Result (initial states): ";
                    result->filter(storm::modelchecker::ExplicitQualitativeCheckResult(model->getInitialStates()));
                    std::cout << *result << std::endl;
                } else {
                    std::cout << " skipped, because the modelling formalism is currently unsupported." << std::endl;
                }

                storm::settings::modules::ParametricSettings const& parametricSettings = storm::settings::parametricSettings();
                if (parametricSettings.exportResultToFile()) {
                    exportParametricResultToFile(result->asExplicitQuantitativeCheckResult<storm::RationalFunction>()[*model->getInitialStates().begin()], storm::models::sparse::Dtmc<storm::RationalFunction>::ConstraintCollector(*(model->template as<storm::models::sparse::Dtmc<storm::RationalFunction>>())), parametricSettings.exportResultPath());
                }
            }
        }
#endif

        template<storm::dd::DdType DdType>
        void verifySymbolicModelWithAbstractionRefinementEngine(storm::prism::Program const& program, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Abstraction Refinement is not yet implemented.");
        }

        template<storm::dd::DdType DdType>
        void verifySymbolicModelWithHybridEngine(std::shared_ptr<storm::models::symbolic::Model<DdType>> model, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            for (auto const& formula : formulas) {
                std::cout << std::endl << "Model checking property: " << *formula << " ...";
                std::unique_ptr<storm::modelchecker::CheckResult> result(storm::verifySymbolicModelWithHybridEngine(model, formula));

                if (result) {
                    std::cout << " done." << std::endl;
                    std::cout << "Result (initial states): ";
                    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<DdType>(model->getReachableStates(), model->getInitialStates()));
                    std::cout << *result << std::endl;
                } else {
                    std::cout << " skipped, because the modelling formalism is currently unsupported." << std::endl;
                }
            }
        }

        template<storm::dd::DdType DdType>
        void verifySymbolicModelWithSymbolicEngine(std::shared_ptr<storm::models::symbolic::Model<DdType>> model, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            for (auto const& formula : formulas) {
                std::cout << std::endl << "Model checking property: " << *formula << " ...";
                std::unique_ptr<storm::modelchecker::CheckResult> result(storm::verifySymbolicModelWithDdEngine(model, formula));
                if (result) {
                    std::cout << " done." << std::endl;
                    std::cout << "Result (initial states): ";
                    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<DdType>(model->getReachableStates(), model->getInitialStates()));
                    std::cout << *result << std::endl;
                } else {
                    std::cout << " skipped, because the modelling formalism is currently unsupported." << std::endl;
                }
            }
        }
        
#define BRANCH_ON_MODELTYPE(result, model, value_type, dd_type, function, ...) \
    if (model->isSymbolicModel()) { \
        if (model->isOfType(storm::models::ModelType::Dtmc)) { \
            result = function<storm::models::symbolic::Dtmc<dd_type>>(model->as<storm::models::symbolic::Dtmc<dd_type>>(), __VA_ARGS__); \
        } else if (model->isOfType(storm::models::ModelType::Ctmc)) { \
            result = function<storm::models::symbolic::Ctmc<dd_type>>(model->as<storm::models::symbolic::Ctmc<dd_type>>(), __VA_ARGS__); \
        } else if (model->isOfType(storm::models::ModelType::Mdp)) { \
            result = function<storm::models::symbolic::Mdp<dd_type>>(model->as<storm::models::symbolic::Mdp<dd_type>>(), __VA_ARGS__); \
        } else { \
            STORM_LOG_ASSERT(false, "Unknown model type."); \
        } \
    } else { \
        STORM_LOG_ASSERT(model->isSparseModel(), "Unknown model type."); \
        if (model->isOfType(storm::models::ModelType::Dtmc)) { \
            result = function<storm::models::sparse::Dtmc<value_type>>(model->as<storm::models::sparse::Dtmc<value_type>>(), __VA_ARGS__); \
        } else if (model->isOfType(storm::models::ModelType::Ctmc)) { \
            result = function<storm::models::sparse::Ctmc<value_type>>(model->as<storm::models::sparse::Ctmc<value_type>>(), __VA_ARGS__); \
        } else if (model->isOfType(storm::models::ModelType::Mdp)) { \
            result = function<storm::models::sparse::Mdp<value_type>>(model->as<storm::models::sparse::Mdp<value_type>>(), __VA_ARGS__); \
        } else if (model->isOfType(storm::models::ModelType::MarkovAutomaton)) { \
            result = function<storm::models::sparse::MarkovAutomaton<value_type>>(model->as<storm::models::sparse::MarkovAutomaton<value_type>>(), __VA_ARGS__); \
        } else { \
            STORM_LOG_ASSERT(false, "Unknown model type."); \
        } \
    }
    
        template<typename ValueType, storm::dd::DdType LibraryType>
        void buildAndCheckSymbolicModel(storm::prism::Program const& program, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            storm::settings::modules::GeneralSettings const& settings = storm::settings::generalSettings();
            
            if (settings.getEngine() == storm::settings::modules::GeneralSettings::Engine::AbstractionRefinement) {
                verifySymbolicModelWithAbstractionRefinementEngine<LibraryType>(program, formulas);
            } else {
                storm::storage::ModelProgramPair modelProgramPair = buildSymbolicModel<ValueType, LibraryType>(program, formulas);
                STORM_LOG_THROW(modelProgramPair.model != nullptr, storm::exceptions::InvalidStateException, "Model could not be constructed for an unknown reason.");
                
                // Preprocess the model if needed.
                BRANCH_ON_MODELTYPE(modelProgramPair.model, modelProgramPair.model, ValueType, LibraryType, preprocessModel, formulas);
                
                // Print some information about the model.
                modelProgramPair.model->printModelInformationToStream(std::cout);
                
                // Verify the model, if a formula was given.
                if (!formulas.empty()) {
                    // There may be constants of the model appearing in the formulas, so we replace all their occurrences
                    // by their definitions in the translated program.
                    
                    // Start by building a mapping from constants of the (translated) model to their defining expressions.
                    std::map<storm::expressions::Variable, storm::expressions::Expression> constantSubstitution;
                    for (auto const& constant : modelProgramPair.program.getConstants()) {
                        if (constant.isDefined()) {
                            constantSubstitution.emplace(constant.getExpressionVariable(), constant.getExpression());
                        }
                    }
                    
                    std::vector<std::shared_ptr<storm::logic::Formula>> preparedFormulas;
                    for (auto const& formula : formulas) {
                        preparedFormulas.emplace_back(formula->substitute(constantSubstitution));
                    }
                    
                    if (modelProgramPair.model->isSparseModel()) {
                        if(settings.isCounterexampleSet()) {
                            // If we were requested to generate a counterexample, we now do so for each formula.
                            for(auto const& formula : preparedFormulas) {
                                generateCounterexample<ValueType>(program, modelProgramPair.model->as<storm::models::sparse::Model<ValueType>>(), formula);
                            }
                        } else {
                            verifySparseModel<ValueType>(modelProgramPair.model->as<storm::models::sparse::Model<ValueType>>(), preparedFormulas);
                        }
                    } else if (modelProgramPair.model->isSymbolicModel()) {
                        if (settings.getEngine() == storm::settings::modules::GeneralSettings::Engine::Hybrid) {
                            verifySymbolicModelWithHybridEngine(modelProgramPair.model->as<storm::models::symbolic::Model<LibraryType>>(), preparedFormulas);
                        } else {
                            verifySymbolicModelWithSymbolicEngine(modelProgramPair.model->as<storm::models::symbolic::Model<LibraryType>>(), preparedFormulas);
                        }
                    } else {
                        STORM_LOG_THROW(false, storm::exceptions::InvalidSettingsException, "Invalid input model type.");
                    }
                }
            }
        }
        
        template<typename ValueType>
        void buildAndCheckSymbolicModel(storm::prism::Program const& program, std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            if (storm::settings::generalSettings().getDdLibraryType() == storm::dd::DdType::CUDD) {
                buildAndCheckSymbolicModel<ValueType, storm::dd::DdType::CUDD>(program, formulas);
            } else if (storm::settings::generalSettings().getDdLibraryType() == storm::dd::DdType::Sylvan) {
                buildAndCheckSymbolicModel<ValueType, storm::dd::DdType::Sylvan>(program, formulas);
            }
        }

        template<typename ValueType>
        void buildAndCheckExplicitModel(std::vector<std::shared_ptr<storm::logic::Formula>> const& formulas) {
            storm::settings::modules::GeneralSettings const& settings = storm::settings::generalSettings();

            STORM_LOG_THROW(settings.isExplicitSet(), storm::exceptions::InvalidStateException, "Unable to build explicit model without model files.");
            std::shared_ptr<storm::models::ModelBase> model = buildExplicitModel<ValueType>(settings.getTransitionFilename(), settings.getLabelingFilename(), settings.isStateRewardsSet() ? settings.getStateRewardsFilename() : boost::optional<std::string>(), settings.isTransitionRewardsSet() ? settings.getTransitionRewardsFilename() : boost::optional<std::string>(), settings.isChoiceLabelingSet() ? settings.getChoiceLabelingFilename() : boost::optional<std::string>());
            
            // Preprocess the model if needed.
            BRANCH_ON_MODELTYPE(model, model, ValueType, storm::dd::DdType::CUDD, preprocessModel, formulas);

            // Print some information about the model.
            model->printModelInformationToStream(std::cout);

            // Verify the model, if a formula was given.
            if (!formulas.empty()) {
                STORM_LOG_THROW(model->isSparseModel(), storm::exceptions::InvalidStateException, "Expected sparse model.");
                verifySparseModel<ValueType>(model->as<storm::models::sparse::Model<ValueType>>(), formulas);
            }
        }
    }
}

#endif //STORM_ENTRYPOINTS_H_H