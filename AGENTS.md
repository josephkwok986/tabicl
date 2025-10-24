# Repository Guidelines

## Project Structure & Module Organization
TabICL is packaged from `src/tabicl`, which exposes the public API through `__init__.py` and groups implementation details under `model/`, `prior/`, `sklearn/`, and `train/`. Unit tests live in `tests/` and mirror the package layout; add new files next to the code they exercise (e.g., `tests/test_model.py`). Training and experiment scripts are kept in `scripts/` (stage1-3 curriculum helpers) while research notebooks and larger assets sit in `brepnet/`, `infer_server/`, and `figures/`. Keep auxiliary utilities inside `my_scripts*/` unless they graduate into the core library.

## Build, Test, and Development Commands
Install editable deps with `pip install -e .` from the repo root. Use Hatch for repeatable workflows:
```bash
hatch run pytest            # run the test suite
hatch run types:check       # mypy type checks (installs stubs if missing)
hatch build                 # produce sdists/wheels before publishing
```
When iterating quickly you can call `pytest tests/test_sklearn.py -k "<pattern>"` to target specific cases.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indents, 88-character lines where practical, and prefer type hints for public APIs. Modules and filenames stay `snake_case`; classes use `PascalCase`; functions, methods, and variables use `snake_case`; constants are `UPPER_SNAKE_CASE`. Preserve the existing docstring tone—short, action-oriented summaries with parameter details when behavior is non-trivial. Use dataclasses or TypedDicts where they improve readability, and keep imports well-grouped (stdlib, third-party, local).

## Testing Guidelines
Pytest drives validation with branch-aware coverage configured in `pyproject.toml`; extend coverage-friendly patterns (avoid dead code, prefer dependency injection). Name tests `test_*` and mirror the behavior being asserted (e.g., `test_classifier_handles_missing_values`). Include GPU-related checks behind guard clauses so the suite passes on CPU-only runners. Run `hatch run pytest` before submitting and regenerate fixtures rather than hard-coding large blobs.

## Commit & Pull Request Guidelines
Recent history follows a Conventional Commits flavor (`fix (Module): message`, `bump: message`). Match that casing and include a focused scope when possible. For pull requests, provide: 1) a concise summary, 2) links to issues or discussion threads, 3) test evidence (`hatch run pytest` output or screenshots for notebooks), and 4) rollout or checkpoint notes when touching pretrained assets. Request reviews from domain owners of affected modules and wait for CI (GitHub Actions `testing.yml`) to pass before merging.
