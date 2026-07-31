"""
gottlux.run — headless orchestration and provenance.

* :func:`~gottlux.run.pipeline.run_path` / :func:`~gottlux.run.pipeline.run_recording`
  run a set of analyses and write a reproducible run folder.
* :class:`~gottlux.run.provenance.RunFolder` builds that folder (manifest + source snapshot).
* :mod:`~gottlux.run.export_provenance` does the same for a GUI export: every video and
  event export lands in a folder holding the artifact, a README naming every source clip,
  and ``provenance.json``.
"""
from __future__ import annotations

from gottlux.run.export_provenance import export_folder, source_facts, write_provenance
from gottlux.run.pipeline import run_path, run_recording
from gottlux.run.provenance import RunFolder

__all__ = ["run_path", "run_recording", "RunFolder",
           "export_folder", "source_facts", "write_provenance"]
