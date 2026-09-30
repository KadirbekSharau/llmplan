"""JSON rendering of a `Timeline` (M5_DESIGN.md section 2).

Every `Timeline` field, in model order, indented like the other JSON outputs. A timeline
holds no wall-clock values, so identical inputs give byte-identical output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmplan.simulate import Timeline


def timeline_json(timeline: Timeline) -> str:
    """Serialize `timeline` as indented JSON with a trailing newline."""
    return json.dumps(timeline.model_dump(mode="json"), indent=2) + "\n"
