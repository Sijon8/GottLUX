# Contributing to GottLUX

GottLUX is research-grade scientific software intended for open release to the event-based-sensing
community. This guide is the **development process** — how the project is structured, the standards
code is held to, and the branching / changelog / release discipline that keeps it professional and
reproducible. It applies equally to the maintainer and to outside contributors.

---

## 1. Repository structure

```
gottlux/                package (importing it pulls in NumPy-level code only)
  config.py             one superset Config — the source of truth for a run's parameters
  sensors.py            sensor/camera profile registry (hardware datasheets + optics)
  cli.py  __main__.py   the `gottlux` entry point (no path → GUI, path → headless)
  io/                   decode → Recording, memmap cache, export, telemetry
  core/                 the math: accumulate, frequency, geometry, photogrammetry,
                        performance (KPIs), tonemap, metrics, denoise
  detectors/            Detector/Param/Target framework + flutter detector + signatures
  rotation/             the rotating-payload stack: de-rotation, fusion, trackers, viz
  viz/                  publication figures (matplotlib, imported lazily)
  app/                  the PySide6 GUI (tabs, multi-clip, range lab) — imported lazily
  run/                  pipeline, provenance/run-folder, reports
tests/                  pytest suite on synthetic data (no private files needed)
docs/                   architecture, algorithms, guides, design history, future work
examples/               runnable quickstart + bundled sample recordings (examples/data/)
```

**Layering rule.** Dependencies point *downward*: `app` → `run` → `detectors`/`viz` → `core` →
`io` → `config`/`sensors`. `core` is pure NumPy (no Qt, no matplotlib at import time). Keep heavy
imports (`PySide6`, `matplotlib`, `numba`, `pandas`) **lazy** — inside the function that needs them
— so `import gottlux` stays light and headless.

## 2. Development setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[all,dev]"                           # editable install + every optional extra
pytest                                                # the synthetic-data suite (no real files)
```

## 3. Coding standards

- **Python ≥ 3.10**, 4-space indent, **100-column** lines (enforced by Ruff — see §7).
- **Type hints** on public function signatures; `from __future__ import annotations` at the top.
- **Docstrings are first-class.** Every module opens with a docstring explaining *what it is and
  why*; public functions/classes explain the *meaning* of parameters, not just their types. Match
  the literate, explain-the-physics style already in the codebase — a reader should learn the model
  from the source.
- **Match surrounding style.** New code should read like the file it lives in (naming, comment
  density, idiom).
- **No project-specific names in the package.** GottLUX is generic event-based-sensor software;
  parameterize hardware through `sensors.py`, never hard-code a deployment's constants.
- **Reproducibility & robustness.** Multi-output analyses compute and save each product
  independently with isolated failures (see `run/performance_report.py` for the `_guard` /
  `_SaveLedger` pattern) so one bad result never loses the rest, and every run archives its config,
  input checksums, environment and a source snapshot.

## 4. Testing

- Tests live in `tests/`, use **synthetic data** (`gottlux.synthetic`) so the suite runs anywhere
  with no private captures, and must pass before any merge.
- Add or update tests for every behavior change. Cover the math at the unit level and the
  command/bundle at the integration level (mirror `tests/test_performance.py`).
- GUI tests run **offscreen** (`QT_QPA_PLATFORM=offscreen`); figure tests use the `Agg` backend.
- Run `pytest` locally; CI runs the same matrix (§7).

## 5. Branching model — *a branch per change* (GitHub Flow)

`main` is **always releasable** and is the only long-lived branch. All work happens on a
short-lived branch off `main`, named by intent:

| prefix | for |
|---|---|
| `feat/…`  | a new capability (e.g. `feat/kpi-metrics`) |
| `fix/…`   | a bug fix |
| `docs/…`  | documentation only |
| `chore/…` | tooling, packaging, CI, housekeeping |
| `refactor/…` | behavior-preserving restructuring |

Workflow for **each major update**:

```bash
git switch main && git pull                 # start from the latest main
git switch -c feat/<short-name>             # branch per change
# … commit in small, logical steps …
pytest                                       # green before opening a PR
git push -u origin feat/<short-name>         # open a Pull Request → review → merge to main
```

- Keep branches focused and short-lived; rebase or merge `main` in if it moves underneath.
- Open a **Pull Request** even when working solo — it’s the review checkpoint and the record of
  *why* a change was made. Squash-merge to keep `main`’s history one-commit-per-change.
- Delete the branch after merge.

**Commit messages:** imperative subject ≤ ~72 chars (`Add prop-frequency range metric`), a blank
line, then the *why*. Reference issues/PRs where relevant.

## 6. Changelog & releases (Semantic Versioning)

- The project follows [Keep a Changelog](https://keepachangelog.com) and
  [SemVer](https://semver.org): **MAJOR**.**MINOR**.**PATCH**.
- **Every user-facing change updates `CHANGELOG.md`** under the top `## [Unreleased]` section
  (`Added` / `Changed` / `Fixed` / `Removed`) in the same PR that makes the change.
- The version is single-sourced from `gottlux/__init__.py` (`__version__`); `pyproject.toml` reads
  it dynamically — bump it in one place.

**Cutting a release:**

1. On a `chore/release-X.Y.Z` branch: bump `__version__`, and rename `## [Unreleased]` to
   `## [X.Y.Z] — <date>` (start a fresh empty `Unreleased`).
2. PR → merge to `main`.
3. Tag and build:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z — <headline>"
   git push origin main --tags
   python -m build                 # sdist + wheel in dist/
   ```

## 7. Continuous integration & tooling

`.github/workflows/ci.yml` runs on every push and PR:

- **Tests** — `pytest` on Python 3.10–3.13 on Linux, and on Python 3.13 on Windows.
- **Lint & format** — `ruff check .` and `ruff format --check .` (config in `pyproject.toml`).

Run them locally before pushing:

```bash
pip install ruff
ruff format .            # auto-format
ruff check --fix .       # lint + safe autofixes
pytest
```

## 8. Documentation

- **Code-level:** module + function docstrings (the literate style above).
- **Project-level (`docs/`):** `ARCHITECTURE.md` (module map), `ALGORITHMS.md` (the math),
  `DETECTOR_GUIDE.md` / `GUI_GUIDE.md` (how-to), `DESIGN_HISTORY.md` (narrative), and
  `FUTURE_WORK.md` (the open roadmap; every item there is available to work on).
- Update the relevant doc **in the same PR** as the change. A new analysis/metric gets: a
  docstring, a README mention, an `ALGORITHMS.md` entry, a test, and a `CHANGELOG.md` line.

---

Contributions are released under the project’s [MIT License](LICENSE).
