#include "storm/solver/TopologicalLinearEquationSolver.h"

#include "storm/environment/solver/TopologicalSolverEnvironment.h"

#include "storm/utility/constants.h"
#include "storm/utility/vector.h"
#include "storm/exceptions/InvalidStateException.h"
#include "storm/exceptions/InvalidEnvironmentException.h"
#include "storm/exceptions/UnexpectedException.h"

namespace storm {
    namespace solver {

        template<typename ValueType>
        TopologicalLinearEquationSolver<ValueType>::TopologicalLinearEquationSolver() : localA(nullptr), A(nullptr) {
            // Intentionally left empty.
        }

        template<typename ValueType>
        TopologicalLinearEquationSolver<ValueType>::TopologicalLinearEquationSolver(storm::storage::SparseMatrix<ValueType> const& A) : localA(nullptr), A(nullptr) {
            this->setMatrix(A);
        }

        template<typename ValueType>
        TopologicalLinearEquationSolver<ValueType>::TopologicalLinearEquationSolver(storm::storage::SparseMatrix<ValueType>&& A) : localA(nullptr), A(nullptr) {
            this->setMatrix(std::move(A));
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::setMatrix(storm::storage::SparseMatrix<ValueType> const& A) {
            localA.reset();
            this->A = &A;
            clearCache();
        }

        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::setMatrix(storm::storage::SparseMatrix<ValueType>&& A) {
            localA = std::make_unique<storm::storage::SparseMatrix<ValueType>>(std::move(A));
            this->A = localA.get();
            clearCache();
        }
        
        template<typename ValueType>
        storm::Environment TopologicalLinearEquationSolver<ValueType>::getEnvironmentForUnderlyingSolver(storm::Environment const& env, bool adaptPrecision) const {
            storm::Environment subEnv(env);
            subEnv.solver().setLinearEquationSolverType(env.solver().topological().getUnderlyingEquationSolverType(), env.solver().topological().isUnderlyingEquationSolverTypeSetFromDefault());
            if (adaptPrecision) {
                STORM_LOG_ASSERT(this->longestSccChainSize, "Did not compute the longest SCC chain size although it is needed.");
                auto subEnvPrec = subEnv.solver().getPrecisionOfLinearEquationSolver(subEnv.solver().getLinearEquationSolverType());
                subEnv.solver().setLinearEquationSolverPrecision(static_cast<storm::RationalNumber>(subEnvPrec.first.get() / storm::utility::convertNumber<storm::RationalNumber>(this->longestSccChainSize.get())));
            }
            return subEnv;
        }

        template<typename ValueType>
        bool TopologicalLinearEquationSolver<ValueType>::internalSolveEquations(Environment const& env, std::vector<ValueType>& x, std::vector<ValueType> const& b) const {
            
            // For sound computations we need to increase the precision in each SCC
            bool needAdaptPrecision = env.solver().isForceSoundness() && env.solver().getPrecisionOfLinearEquationSolver(env.solver().topological().getUnderlyingEquationSolverType()).first.is_initialized();
            
            if (!this->sortedSccDecomposition || (needAdaptPrecision && !this->longestSccChainSize)) {
                STORM_LOG_TRACE("Creating SCC decomposition.");
                createSortedSccDecomposition(needAdaptPrecision);
            }
            
            //std::cout << "Sorted SCC decomposition: " << std::endl;
            //for (auto const& scc : *this->sortedSccDecomposition) {
                //std::cout << "SCC: ";
              //  for (auto const& row : scc) {
                    //std::cout << row << "  ";
               // }
                //std::cout << std::endl;
            //}
            
            // We do not need to adapt the precision if all SCCs are trivial (i.e., the system is acyclic)
            needAdaptPrecision = needAdaptPrecision && (this->sortedSccDecomposition->size() != this->getMatrixRowCount());
            
            storm::Environment sccSolverEnvironment = getEnvironmentForUnderlyingSolver(env, needAdaptPrecision);
            
            std::cout << "Found " << this->sortedSccDecomposition->size() << "SCCs. Average size is " << static_cast<double>(this->getMatrixRowCount()) / static_cast<double>(this->sortedSccDecomposition->size()) << "." << std::endl;
            if (this->longestSccChainSize) {
                std::cout << "Longest SCC chain size is " << this->longestSccChainSize.get() << std::endl;
            }
            
            // Handle the case where there is just one large SCC
            bool returnValue = true;
            if (this->sortedSccDecomposition->size() == 1) {
                returnValue = solveFullyConnectedEquationSystem(sccSolverEnvironment, x, b);
            } else {
                storm::storage::BitVector sccAsBitVector(x.size(), false);
                for (auto const& scc : *this->sortedSccDecomposition) {
                    if (scc.isTrivial()) {
                        returnValue = solveTrivialScc(*scc.begin(), x, b) && returnValue;
                        ++this->overallPerformedIterations;
                    } else {
                        sccAsBitVector.clear();
                        for (auto const& state : scc) {
                            sccAsBitVector.set(state, true);
                        }
                        returnValue = solveScc(sccSolverEnvironment, sccAsBitVector, x, b) && returnValue;
                    }
                }
            }
            
            if (this->sccSolver) {
                this->overallPerformedIterations += this->sccSolver->overallPerformedIterations;
                this->sccSolver->overallPerformedIterations = 0;
            }
            if (!this->isCachingEnabled()) {
                clearCache();
            }

            

            
            return returnValue;
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::createSortedSccDecomposition(bool needLongestChainSize) const {
            // Obtain the scc decomposition
            auto sccDecomposition = storm::storage::StronglyConnectedComponentDecomposition<ValueType>(*this->A);
            
            // Get a mapping from matrix row to the corresponding scc
            STORM_LOG_THROW(sccDecomposition.size() < std::numeric_limits<uint32_t>::max(), storm::exceptions::UnexpectedException, "The number of SCCs is too large.");
            std::vector<uint32_t> sccIndices(this->A->getRowCount(), std::numeric_limits<uint32_t>::max());
            uint32_t sccIndex = 0;
            for (auto const& scc : sccDecomposition) {
                for (auto const& row : scc) {
                    sccIndices[row] = sccIndex;
                }
                ++sccIndex;
            }
            
            // Prepare the resulting set of sorted sccs
            this->sortedSccDecomposition = std::make_unique<std::vector<storm::storage::StronglyConnectedComponent>>();
            std::vector<storm::storage::StronglyConnectedComponent>& sortedSCCs = *this->sortedSccDecomposition;
            sortedSCCs.reserve(sccDecomposition.size());
            
            // Find a topological sort via DFS.
            storm::storage::BitVector unsortedSCCs(sccDecomposition.size(), true);
            std::vector<uint32_t> sccStack, chainSizes;
            if (needLongestChainSize) {
                chainSizes.resize(sccDecomposition.size(), 1u);
            }
            uint32_t longestChainSize = 0;
            uint32_t const token = std::numeric_limits<uint32_t>::max();
            std::set<uint64_t> successorSCCs;

            for (uint32_t firstUnsortedScc = 0; firstUnsortedScc < unsortedSCCs.size(); firstUnsortedScc = unsortedSCCs.getNextSetIndex(firstUnsortedScc + 1)) {
                
                sccStack.push_back(firstUnsortedScc);
                while (!sccStack.empty()) {
                    uint32_t currentSccIndex = sccStack.back();
                    if (currentSccIndex != token) {
                        // Check whether the SCC is still unprocessed
                        if (unsortedSCCs.get(currentSccIndex)) {
                            // Explore the successors of the scc.
                            storm::storage::StronglyConnectedComponent const& currentScc = sccDecomposition.getBlock(currentSccIndex);
                            // We first push a token on the stack in order to recognize later when all successors of this SCC have been explored already.
                            sccStack.push_back(token);
                            // Now add all successors that are not already sorted.
                            // Successors should only be added once, so we first prepare a set of them and add them afterwards.
                            successorSCCs.clear();
                            for (auto const& row : currentScc) {
                                for (auto const& entry : this->A->getRow(row)) {
                                    auto const& successorSCC = sccIndices[entry.getColumn()];
                                    if (successorSCC != currentSccIndex && unsortedSCCs.get(successorSCC)) {
                                        successorSCCs.insert(successorSCC);
                                    }
                                }
                            }
                            sccStack.insert(sccStack.end(), successorSCCs.begin(), successorSCCs.end());
                            
                        }
                    } else {
                        // all successors of the current scc have already been explored.
                        sccStack.pop_back(); // pop the token
                        
                        currentSccIndex = sccStack.back();
                        storm::storage::StronglyConnectedComponent& scc = sccDecomposition.getBlock(currentSccIndex);
                        
                        // Compute the longest chain size for this scc
                        if (needLongestChainSize) {
                            uint32_t& currentChainSize = chainSizes[currentSccIndex];
                            for (auto const& row : scc) {
                                for (auto const& entry : this->A->getRow(row)) {
                                    auto const& successorSCC = sccIndices[entry.getColumn()];
                                    if (successorSCC != currentSccIndex) {
                                        currentChainSize = std::max(currentChainSize, chainSizes[successorSCC] + 1);
                                    }
                                }
                            }
                            longestChainSize = std::max(longestChainSize, currentChainSize);
                        }
                        
                        unsortedSCCs.set(currentSccIndex, false);
                        sccStack.pop_back(); // pop the current scc index
                        sortedSCCs.push_back(std::move(scc));
                    }
                }
            }
            
            if (longestChainSize > 0) {
                this->longestSccChainSize = longestChainSize;
            }
        }
        
        template<typename ValueType>
        bool TopologicalLinearEquationSolver<ValueType>::solveTrivialScc(uint64_t const& sccState, std::vector<ValueType>& globalX, std::vector<ValueType> const& globalB) const {
            ValueType& xi = globalX[sccState];
            xi = globalB[sccState];
            bool hasDiagonalEntry = false;
            ValueType denominator;
            for (auto const& entry : this->A->getRow(sccState)) {
                if (entry.getColumn() == sccState) {
                    STORM_LOG_ASSERT(!storm::utility::isOne(entry.getValue()), "Diagonal entry of fix point system has value 1.");
                    hasDiagonalEntry = true;
                    denominator = storm::utility::one<ValueType>() - entry.getValue();
                } else {
                    xi += entry.getValue() * globalX[entry.getColumn()];
                }
            }
            
            if (hasDiagonalEntry) {
                xi /= denominator;
            }
            //std::cout << "Solved trivial scc " << sccState << " with result " << globalX[sccState] << std::endl;
            return true;
        }
        
        template<typename ValueType>
        bool TopologicalLinearEquationSolver<ValueType>::solveFullyConnectedEquationSystem(storm::Environment const& sccSolverEnvironment, std::vector<ValueType>& x, std::vector<ValueType> const& b) const {
            if (!this->sccSolver) {
                this->sccSolver = GeneralLinearEquationSolverFactory<ValueType>().create(sccSolverEnvironment, LinearEquationSolverTask::SolveEquations);
                this->sccSolver->setCachingEnabled(true);
                this->sccSolver->setBoundsFromOtherSolver(*this);
                if (this->sccSolver->getEquationProblemFormat(sccSolverEnvironment) == LinearEquationSolverProblemFormat::EquationSystem) {
                    // Convert the matrix to an equation system. Note that we need to insert diagonal entries.
                    storm::storage::SparseMatrix<ValueType> eqSysA(*this->A, true);
                    eqSysA.convertToEquationSystem();
                    this->sccSolver->setMatrix(std::move(eqSysA));
                } else {
                    this->sccSolver->setMatrix(*this->A);
                }
            }
            return this->sccSolver->solveEquations(sccSolverEnvironment, x, b);
        }
        
        template<typename ValueType>
        bool TopologicalLinearEquationSolver<ValueType>::solveScc(storm::Environment const& sccSolverEnvironment, storm::storage::BitVector const& scc, std::vector<ValueType>& globalX, std::vector<ValueType> const& globalB) const {
            
            // Set up the SCC solver
            if (!this->sccSolver) {
                this->sccSolver = GeneralLinearEquationSolverFactory<ValueType>().create(sccSolverEnvironment, LinearEquationSolverTask::SolveEquations);
                this->sccSolver->setCachingEnabled(true);
            }
            
            // Matrix
            bool asEquationSystem = this->sccSolver->getEquationProblemFormat(sccSolverEnvironment) == LinearEquationSolverProblemFormat::EquationSystem;
            storm::storage::SparseMatrix<ValueType> sccA = this->A->getSubmatrix(true, scc, scc, asEquationSystem);
            if (asEquationSystem) {
                sccA.convertToEquationSystem();
            }
            //std::cout << "Solving SCC " << scc << std::endl;
            //std::cout << "Matrix is " << sccA << std::endl;
            this->sccSolver->setMatrix(std::move(sccA));
            
            // x Vector
            auto sccX = storm::utility::vector::filterVector(globalX, scc);
            
            // b Vector
            std::vector<ValueType> sccB;
            sccB.reserve(scc.getNumberOfSetBits());
            for (auto const& row : scc) {
                ValueType bi = globalB[row];
                for (auto const& entry : this->A->getRow(row)) {
                    if (!scc.get(entry.getColumn())) {
                        bi += entry.getValue() * globalX[entry.getColumn()];
                    }
                }
                sccB.push_back(std::move(bi));
            }
            
            // lower/upper bounds
            if (this->hasLowerBound(storm::solver::AbstractEquationSolver<ValueType>::BoundType::Global)) {
                this->sccSolver->setLowerBound(this->getLowerBound());
            } else if (this->hasLowerBound(storm::solver::AbstractEquationSolver<ValueType>::BoundType::Local)) {
                this->sccSolver->setLowerBounds(storm::utility::vector::filterVector(this->getLowerBounds(), scc));
            }
            if (this->hasUpperBound(storm::solver::AbstractEquationSolver<ValueType>::BoundType::Global)) {
                this->sccSolver->setUpperBound(this->getUpperBound());
            } else if (this->hasUpperBound(storm::solver::AbstractEquationSolver<ValueType>::BoundType::Local)) {
                this->sccSolver->setUpperBounds(storm::utility::vector::filterVector(this->getUpperBounds(), scc));
            }
            
            //std::cout << "rhs is " << storm::utility::vector::toString(sccB) << std::endl;
            //std::cout << "x is " << storm::utility::vector::toString(sccX) << std::endl;
            
            bool returnvalue = this->sccSolver->solveEquations(sccSolverEnvironment, sccX, sccB);
            storm::utility::vector::setVectorValues(globalX, scc, sccX);
            return returnvalue;
        }
        
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::multiply(std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<ValueType>& result) const {
            if (&x != &result) {
                multiplier.multAdd(*A, x, b, result);
            } else {
                // If the two vectors are aliases, we need to create a temporary.
                if (!this->cachedRowVector) {
                    this->cachedRowVector = std::make_unique<std::vector<ValueType>>(getMatrixRowCount());
                }
                
                multiplier.multAdd(*A, x, b, *this->cachedRowVector);
                result.swap(*this->cachedRowVector);
                
                if (!this->isCachingEnabled()) {
                    clearCache();
                }
            }
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::multiplyAndReduce(OptimizationDirection const& dir, std::vector<uint64_t> const& rowGroupIndices, std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<ValueType>& result, std::vector<uint_fast64_t>* choices) const {
            if (&x != &result) {
                multiplier.multAddReduce(dir, rowGroupIndices, *A, x, b, result, choices);
            } else {
                // If the two vectors are aliases, we need to create a temporary.
                if (!this->cachedRowVector) {
                    this->cachedRowVector = std::make_unique<std::vector<ValueType>>(getMatrixRowCount());
                }
            
                multiplier.multAddReduce(dir, rowGroupIndices, *A, x, b, *this->cachedRowVector, choices);
                result.swap(*this->cachedRowVector);
                
                if (!this->isCachingEnabled()) {
                    clearCache();
                }
            }
        }
        
        template<typename ValueType>
        bool TopologicalLinearEquationSolver<ValueType>::supportsGaussSeidelMultiplication() const {
            return true;
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::multiplyGaussSeidel(std::vector<ValueType>& x, std::vector<ValueType> const* b) const {
            STORM_LOG_ASSERT(this->A->getRowCount() == this->A->getColumnCount(), "This function is only applicable for square matrices.");
            multiplier.multAddGaussSeidelBackward(*A, x, b);
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::multiplyAndReduceGaussSeidel(OptimizationDirection const& dir, std::vector<uint64_t> const& rowGroupIndices, std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<uint_fast64_t>* choices) const {
            multiplier.multAddReduceGaussSeidelBackward(dir, rowGroupIndices, *A, x, b, choices);
        }
        
        template<typename ValueType>
        ValueType TopologicalLinearEquationSolver<ValueType>::multiplyRow(uint64_t const& rowIndex, std::vector<ValueType> const& x) const {
            return multiplier.multiplyRow(*A, rowIndex, x);
        }
        
        template<typename ValueType>
        LinearEquationSolverProblemFormat TopologicalLinearEquationSolver<ValueType>::getEquationProblemFormat(Environment const& env) const {
            return LinearEquationSolverProblemFormat::FixedPointSystem;
        }
        
        template<typename ValueType>
        LinearEquationSolverRequirements TopologicalLinearEquationSolver<ValueType>::getRequirements(Environment const& env, LinearEquationSolverTask const& task) const {
            // Return the requirements of the underlying solver
            return GeneralLinearEquationSolverFactory<ValueType>().getRequirements(getEnvironmentForUnderlyingSolver(env), task);
        }
        
        template<typename ValueType>
        void TopologicalLinearEquationSolver<ValueType>::clearCache() const {
            sortedSccDecomposition.reset();
            longestSccChainSize = boost::none;
            sccSolver.reset();
            LinearEquationSolver<ValueType>::clearCache();
        }
        
        template<typename ValueType>
        uint64_t TopologicalLinearEquationSolver<ValueType>::getMatrixRowCount() const {
            return this->A->getRowCount();
        }
        
        template<typename ValueType>
        uint64_t TopologicalLinearEquationSolver<ValueType>::getMatrixColumnCount() const {
            return this->A->getColumnCount();
        }
        
        template<typename ValueType>
        std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> TopologicalLinearEquationSolverFactory<ValueType>::create(Environment const& env, LinearEquationSolverTask const& task) const {
            return std::make_unique<storm::solver::TopologicalLinearEquationSolver<ValueType>>();
        }
        
        template<typename ValueType>
        std::unique_ptr<LinearEquationSolverFactory<ValueType>> TopologicalLinearEquationSolverFactory<ValueType>::clone() const {
            return std::make_unique<TopologicalLinearEquationSolverFactory<ValueType>>(*this);
        }
        
        // Explicitly instantiate the linear equation solver.
        template class TopologicalLinearEquationSolver<double>;
        template class TopologicalLinearEquationSolverFactory<double>;
        
#ifdef STORM_HAVE_CARL
        template class TopologicalLinearEquationSolver<storm::RationalNumber>;
        template class TopologicalLinearEquationSolverFactory<storm::RationalNumber>;
        
        template class TopologicalLinearEquationSolver<storm::RationalFunction>;
        template class TopologicalLinearEquationSolverFactory<storm::RationalFunction>;

#endif
    }
}
