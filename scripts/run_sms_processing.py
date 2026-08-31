#!/usr/bin/env python3
"""Run the aggregate-safe local SMS processing CLI from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pocketfinancer_sms.cli import main  # noqa: E402


raise SystemExit(main())
