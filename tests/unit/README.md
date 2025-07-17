# Unit Tests

## Purpose

Unit tests focus on testing **individual components in isolation**. These tests should be:
- **Fast** (< 100ms per test typically)
- **Independent** (no external dependencies like MPI, network, or file system)
- **Focused** (testing a single function, method, or class)
- **Deterministic** (same input always produces same output)

## What Goes Here

### ✅ **Good Examples:**
- Testing configuration validation logic
- Testing data structure initialization  
- Testing utility functions and methods
- Testing error handling for invalid inputs
- Testing class instantiation and basic properties
- Testing pure computational logic

### ❌ **What Doesn't Belong:**
- Tests requiring MPI communication between processes
- Tests requiring GPU/CUDA functionality
- Tests requiring file I/O or network access
- Tests that depend on specific hardware availability
- Tests that require multiple components working together

## Structure

Each component should have its own subdirectory:

```
unit/
├── communicator/          # Communication layer tests
│   ├── test_basic.py      # Basic functionality
│   ├── test_config.py     # Configuration validation
│   └── test_backend_selection.py  # Backend selection logic
├── solvers/               # Solver component tests
├── kernels/               # Kernel computation tests
└── models/                # Model definition tests
```

## Running Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run unit tests for specific component
python -m pytest tests/unit/communicator/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=src/dalia --cov-report=html
```

## Guidelines

1. **Use mocks** for external dependencies (files, network, MPI)
2. **Test edge cases** and error conditions
3. **Keep tests small** and focused on single functionality
4. **Use descriptive test names** that explain what is being tested
5. **Follow AAA pattern**: Arrange, Act, Assert
6. **Mock backend_flags** when testing availability logic

## Example Test Structure

```python
class TestComponentBasic:
    """Basic functionality tests for Component."""
    
    def test_component_initialization_with_valid_config(self):
        """Test that component initializes correctly with valid configuration."""
        # Arrange
        config = ComponentConfig(param="valid_value")
        
        # Act
        component = Component(config)
        
        # Assert
        assert component.param == "valid_value"
        assert component.is_initialized
    
    @pytest.mark.parametrize("invalid_param", ["", None, 123])
    def test_component_raises_error_with_invalid_config(self, invalid_param):
        """Test that component raises appropriate errors for invalid configuration."""
        with pytest.raises(ValueError, match="Invalid parameter"):
            ComponentConfig(param=invalid_param)
```
