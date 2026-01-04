# Contributing to Reference-to-Rig

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository
2. Run the setup script:
   - Windows: `scripts\setup.bat`
   - macOS/Linux: `bash scripts/setup.sh`

## Code Style

### Python (Engine)

- Format with `black` (100 char line length)
- Lint with `ruff`
- Type hints required for all public functions
- Follow existing patterns in the codebase

Run before committing:
```bash
cd engine
source venv/bin/activate  # or venv\Scripts\activate on Windows
black app tests scripts
ruff check --fix app tests scripts
```

### TypeScript (UI)

- Use TypeScript strict mode
- Format with Prettier
- Follow React best practices
- Use functional components with hooks

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests if applicable
4. Update documentation if needed
5. Run linters and tests
6. Submit PR with clear description

## Testing

### Running Tests

```bash
cd engine
pytest -v
```

### Adding Tests

- Add tests for new features
- Test file naming: `test_*.py`
- Use pytest fixtures for common setup

## Architecture Guidelines

### Engine

- Keep modules focused and single-purpose
- Use dependency injection for backends
- All heavy work via background tasks
- Structured logging with correlation IDs

### UI

- Use Zustand for state management
- Keep API calls in `api/client.ts`
- Use Tailwind for styling
- Prefer composition over complex components

## Adding Features

### New Isolation Backend

1. Implement `IsolationBackend` protocol in `app/isolation/adapter.py`
2. Add to `get_isolation_backend()` factory
3. Update `RTR_ISOLATION_BACKEND` config option
4. Add tests

### New NAM Backend

1. Implement `NAMBackend` protocol in `app/rendering/nam_adapter.py`
2. Add to `get_nam_backend()` factory
3. Update `RTR_NAM_BACKEND` config option
4. Add tests

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase

Thank you for contributing!


