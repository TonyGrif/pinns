# Physics-Informed Neural Networks

## Requirements
TBW

## Directory Structure
```
pinns/          # Library Source Code
pyproject.toml  # Project Build File
```

## Development
This project makes use of [uv](https://docs.astral.sh/uv/) for dependency management.
After cloning, install dependencies and set up git hooks using:
```bash
uv sync
uv run poe setup
```

### Additional Dependencies
Run `uv sync` to gather main and development dependencies.

### Pip File Creation
Some HPCs and systems may not have `uv` installed on them: to that end, create
a pip file using `uv run poe pip`.

### Utility Commands
This project contains a couple of utility commands for maintaining consistent code
formatting and quality:

| Command | Description |
|---|---|
| `uv run poe format` | Format and Auto-Fix Lint Issues |
| `uv run poe lint` | Type-Check and Lint |
