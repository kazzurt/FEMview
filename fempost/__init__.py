"""fempost: read-only post-processing viewer for CalculiX runs of the KurtsFEM pipeline.

Ground truth is the CalculiX triple ``<job>.inp`` + ``<job>.frd`` + ``<job>.dat``; sidecar files
written by the KurtsFEM ``structural`` package or by the Codex study scripts only add labels and
summary numbers. Nothing in this package writes beside a run; the only writable location is the
cache under ``structural-fea-environment/temporary/fempost-cache/``. Units are N, mm, MPa.
"""
__version__ = "0.1.0"
