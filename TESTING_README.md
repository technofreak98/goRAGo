# Testing Guide for RAG-ES System

This document provides comprehensive information about testing the RAG-ES (Retrieval-Augmented Generation with Elasticsearch) system.

## Overview

The RAG-ES system includes comprehensive unit tests, integration tests, and end-to-end tests covering all major functionalities:

- **Service Layer**: ElasticsearchService, LLMService, RetrievalService, EmbeddingService
- **Agent Layer**: RAGGraph, LangGraphAgent
- **Router Layer**: Agent Router, Search Router
- **Model Layer**: All Pydantic models
- **Integration Tests**: End-to-end workflows

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests
│   ├── services/              # Service layer tests
│   │   ├── test_elasticsearch_service.py
│   │   ├── test_llm_service.py
│   │   ├── test_retrieval_service.py
│   │   └── test_embedding_service.py
│   ├── agents/                # Agent layer tests
│   │   ├── test_rag_graph.py
│   │   └── test_langgraph_agent.py
│   ├── routers/               # Router layer tests
│   │   ├── test_agent_router.py
│   │   └── test_search_router.py
│   └── models/                # Model layer tests
│       └── test_models.py
├── integration/               # Integration tests
│   └── test_end_to_end.py
└── fixtures/                  # Test fixtures (if needed)
```

## Prerequisites

1. **Conda Environment**: Ensure you're using the `Agentic-RAG` conda environment:
   ```bash
   conda activate Agentic-RAG
   ```

2. **Dependencies**: Install testing dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**: Set up your `.env` file with required API keys:
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

## Running Tests

### Quick Start

Use the provided test runner script:

```bash
python run_tests.py
```

This will run all tests with coverage and generate an HTML coverage report.

### Manual Test Execution

#### Run All Tests
```bash
pytest tests/ -v
```

#### Run Unit Tests Only
```bash
pytest tests/unit/ -v
```

#### Run Integration Tests Only
```bash
pytest tests/integration/ -v
```

#### Run Specific Test File
```bash
pytest tests/unit/services/test_elasticsearch_service.py -v
```

#### Run Specific Test Function
```bash
pytest tests/unit/services/test_elasticsearch_service.py::TestElasticsearchService::test_create_indices_success -v
```

### Test Categories

#### By Markers
```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run slow tests
pytest -m slow -v

# Run external service tests
pytest -m external -v
```

#### By Pattern
```bash
# Run tests matching a pattern
pytest -k "test_elasticsearch" -v

# Run tests excluding a pattern
pytest -k "not test_elasticsearch" -v
```

### Coverage Reports

#### Terminal Coverage
```bash
pytest tests/ --cov=app --cov-report=term-missing
```

#### HTML Coverage Report
```bash
pytest tests/ --cov=app --cov-report=html:htmlcov
```
Open `htmlcov/index.html` in your browser to view the detailed coverage report.

#### Coverage with Threshold
```bash
pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=80
```

### Parallel Test Execution

For faster test execution on multi-core systems:

```bash
pytest tests/ -n auto
```

## Test Configuration

### Pytest Configuration

The `pytest.ini` file contains the test configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=app
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    external: Tests that require external services
```

### Test Fixtures

The `conftest.py` file provides shared fixtures:

- `mock_elasticsearch_client`: Mocked Elasticsearch client
- `mock_openai_client`: Mocked OpenAI client
- `sample_document`: Sample document for testing
- `sample_search_results`: Sample search results
- `sample_search_query`: Sample search query
- `mock_weather_data`: Mock weather data
- `mock_agent_state`: Mock agent state
- `mock_settings`: Mock application settings

## Test Categories Explained

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with mocked dependencies.

**Coverage**:
- Service layer components (ElasticsearchService, LLMService, etc.)
- Agent components (RAGGraph, LangGraphAgent)
- Router components (API endpoints)
- Model validation and serialization

**Key Features**:
- Fast execution
- No external dependencies
- Comprehensive mocking
- Edge case testing

### Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and end-to-end workflows.

**Coverage**:
- Complete query processing workflows
- Service integration
- Error handling across components
- Performance characteristics

**Key Features**:
- Realistic data flow
- Component interaction testing
- Workflow validation

## Mocking Strategy

### External Services

All external services are mocked to ensure:
- **Fast execution**: No network calls
- **Reliability**: Tests don't depend on external service availability
- **Predictability**: Consistent test results
- **Isolation**: Tests focus on business logic

### Mocked Services

1. **Elasticsearch**: Mocked client with realistic responses
2. **OpenAI API**: Mocked client with configurable responses
3. **Weather API**: Mocked weather data responses
4. **File System**: Mocked file operations

## Writing New Tests

### Test File Structure

```python
"""Unit tests for [ComponentName]."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.module.component import ComponentClass


@pytest.mark.unit
class TestComponentClass:
    """Test cases for ComponentClass."""

    @pytest.fixture
    def component(self):
        """Create component instance with mocked dependencies."""
        return ComponentClass()

    @pytest.mark.asyncio
    async def test_method_success(self, component):
        """Test successful method execution."""
        # Arrange
        input_data = "test input"
        expected_output = "expected output"
        
        # Act
        result = await component.method(input_data)
        
        # Assert
        assert result == expected_output

    def test_method_error(self, component):
        """Test method error handling."""
        # Arrange
        invalid_input = None
        
        # Act & Assert
        with pytest.raises(ValueError):
            component.method(invalid_input)
```

### Test Naming Conventions

- **Test files**: `test_<component_name>.py`
- **Test classes**: `Test<ComponentName>`
- **Test methods**: `test_<method_name>_<scenario>`
- **Fixtures**: `<component_name>` or `mock_<service_name>`

### Test Markers

Use appropriate markers for test categorization:

```python
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.external
```

## Continuous Integration

### GitHub Actions (if applicable)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in Python path
2. **Async Test Issues**: Use `@pytest.mark.asyncio` for async tests
3. **Mock Issues**: Verify mock setup and call assertions
4. **Environment Issues**: Check conda environment activation

### Debug Mode

Run tests in debug mode for detailed output:

```bash
pytest tests/ -v -s --tb=long
```

### Test Discovery

If tests aren't being discovered:

```bash
pytest --collect-only tests/
```

## Performance Testing

### Load Testing

For performance testing, consider:

```python
import time
import asyncio

@pytest.mark.slow
async def test_performance():
    """Test component performance."""
    start_time = time.time()
    
    # Perform operations
    await component.operation()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    assert execution_time < 1.0  # Should complete within 1 second
```

### Memory Testing

Monitor memory usage during tests:

```bash
pytest tests/ --profile
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Naming**: Use descriptive test names
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock External Dependencies**: Don't rely on external services
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Maintain Test Data**: Keep test fixtures up to date
7. **Regular Execution**: Run tests frequently during development
8. **Coverage Goals**: Aim for >80% code coverage

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this documentation if needed
5. Add integration tests for new workflows

## Support

For testing-related questions or issues:

1. Check this documentation
2. Review existing test examples
3. Check pytest documentation
4. Create an issue with test details
