# Development Setup

## Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd proloaf
```

### 2. Install the package with development dependencies

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- The `proloaf` package in editable mode
- All runtime dependencies
- Development tools: `ruff` (linter/formatter) and `pre-commit`

### 3. Install pre-commit hooks

```bash
# One-time setup to enable automatic checks on commit
pre-commit install
```

Now the pre-commit hooks will run automatically on every commit.

### Manual pre-commit execution

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks temporarily (use sparingly!)
git commit --no-verify -m "Your message"
```

## Code Style

- **Line length**: 120 characters
- **Python version**: 3.9+ (modern syntax)
- **Formatter**: Ruff (Black-compatible)
- **Import sorting**: Ruff (isort-compatible)
- **Configuration**: See `pyproject.toml` under `[tool.ruff]`

All formatting and linting is handled automatically by pre-commit hooks. See `.pre-commit-guide.md` for more information about the pre-commit setup and Ruff configuration.
