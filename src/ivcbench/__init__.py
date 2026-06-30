"""Public package metadata for ivcbench.

The package is intentionally small at import time. Most users enter through the
Makefile or scripts, while library users can import subpackages such as
``ivcbench.metrics``, ``ivcbench.splits``, and ``ivcbench.eval`` directly.
"""

from __future__ import annotations

__version__ = "1.1.5"

__all__ = ["__version__"]
