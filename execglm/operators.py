"""Operator IDs shared with the DynamicRTL graph format."""
import json
from pathlib import Path
OP = json.loads(Path(__file__).with_name("operator_types.json").read_text())
