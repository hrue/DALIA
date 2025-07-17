# Integration Tests

## Purpose

Integration tests verify **complete end-to-end workflows** involving multiple major components of the DALIA framework. These tests ensure that the entire system works together correctly for real-world use cases.

## What Goes Here

### ✅ **Good Examples:**
- **Complete Gaussian Process workflows** from data loading to inference
- **Full solver pipelines** with real datasets and convergence verification
- **Multi-component interactions** (models + solvers + communicators)
- **Performance benchmarks** on representative problems
- **Regression tests** for scientific accuracy and reproducibility
- **Scalability tests** across different hardware configurations
- **User workflow scenarios** matching actual research use cases

### ❌ **What Doesn't Belong:**
- Single component functionality (→ `unit/` or `component_integration/`)
- Internal component interactions (→ `component_integration/`)
- Tests that can be mocked or isolated (→ `unit/`)

## Testing Pyramid Level

**High complexity, few tests:**
- Most expensive tests (minutes to hours per test)
- Test complete user workflows with real data
- Require full system setup (MPI + GPU + datasets)
- Focus on correctness, convergence, and performance
- Run less frequently (nightly builds, releases)

## Structure

Organize by scientific use case or workflow type:

```
integration/
├── workflows/
│   ├── test_gaussian_process_inference.py    # Complete GP workflow
│   ├── test_spatial_temporal_modeling.py     # Spatio-temporal problems
│   └── test_large_scale_optimization.py      # Scalability testing
├── benchmarks/
│   ├── test_performance_regression.py        # Performance benchmarks
│   ├── test_memory_usage.py                  # Memory efficiency
│   └── test_scaling_behavior.py              # Multi-node scaling
├── examples/
│   ├── test_example_scripts.py               # Validate example code
│   └── test_documentation_examples.py        # Test docs examples
└── regression/
    ├── test_numerical_accuracy.py            # Scientific correctness
    └── test_reproducibility.py               # Deterministic results
```

## Running Integration Tests

```bash
# Run all integration tests (slow!)
python -m pytest tests/integration/ -v

# Run specific workflow
python -m pytest tests/integration/workflows/test_gaussian_process_inference.py -v

# Run with timeout for long tests
python -m pytest tests/integration/ -v --timeout=3600

# Run only fast integration tests
python -m pytest tests/integration/ -v -m "not slow"
```

## Guidelines

1. **Use real datasets** representative of actual use cases
2. **Test convergence** to known solutions when possible
3. **Verify scientific accuracy** against analytical solutions or literature
4. **Test scaling behavior** across different problem sizes
5. **Include timing and memory benchmarks**
6. **Use markers** for different test categories:
   ```python
   @pytest.mark.slow
   @pytest.mark.gpu_required
   @pytest.mark.large_memory
   @pytest.mark.multi_node
   ```
7. **Provide clear failure diagnostics** for complex workflows
8. **Document test requirements** (data files, hardware, time)

## Example Test Structure

```python
class TestGaussianProcessWorkflow:
    """End-to-end tests for complete Gaussian Process workflows."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_size", [1000, 10000])
    def test_complete_gp_inference_workflow(self, dataset_size):
        """Test complete GP inference from data loading to prediction."""
        # Arrange
        data = load_test_dataset(size=dataset_size)
        config = create_gp_config_for_dataset(data)
        
        # Act - Complete workflow
        model = GaussianProcess(config)
        model.fit(data.X_train, data.y_train)
        predictions = model.predict(data.X_test)
        
        # Assert - Scientific correctness
        rmse = compute_rmse(predictions, data.y_test)
        assert rmse < data.expected_rmse, f"RMSE {rmse} exceeds threshold"
        
        # Assert - Performance
        assert model.training_time < data.max_training_time
        assert model.memory_usage < data.max_memory_mb * 1024**2
    
    @pytest.mark.convergence
    def test_solver_convergence_on_known_problem(self):
        """Test solver convergence to analytical solution."""
        # Use a problem with known analytical solution
        problem = create_analytical_test_problem()
        
        solver = Solver(config=problem.solver_config)
        solution = solver.solve(problem)
        
        # Verify convergence to analytical solution
        analytical_solution = problem.analytical_solution()
        error = np.linalg.norm(solution - analytical_solution)
        assert error < 1e-6, f"Solution error {error} too large"
```

## Test Categories

Use pytest markers to categorize tests:

- `@pytest.mark.slow` - Tests taking > 30 seconds
- `@pytest.mark.gpu_required` - Requires GPU hardware
- `@pytest.mark.multi_node` - Requires multiple compute nodes
- `@pytest.mark.large_memory` - Requires > 16GB RAM
- `@pytest.mark.convergence` - Tests numerical convergence
- `@pytest.mark.benchmark` - Performance benchmark tests

## Data Requirements

Integration tests may require:
- **Large datasets** (store in separate data directory)
- **Reference solutions** for validation
- **Multiple hardware configurations** for scaling tests
- **Specific software versions** for reproducibility

Consider using external data repositories or generating synthetic data when datasets are too large for the repository.
