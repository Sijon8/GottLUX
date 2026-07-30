"""
gottlux.run — headless orchestration and provenance.

* :func:`~gottlux.run.pipeline.run_path` / :func:`~gottlux.run.pipeline.run_recording`
  run a set of analyses and write a reproducible run folder.
* :class:`~gottlux.run.provenance.RunFolder` builds that folder (manifest + source snapshot).
"""
from __future__ import annotations

from gottlux.run.pipeline import run_path, run_recording
from gottlux.run.provenance import RunFolder

__all__ = ["run_path", "run_recording", "RunFolder"]
