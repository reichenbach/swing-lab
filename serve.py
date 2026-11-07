#!/usr/bin/env python
"""Run the swing_lab web app locally.

    PYTHONPATH=. .venv/bin/python serve.py
    # then open http://127.0.0.1:5000

Upload a swing, get the report, browse each player's history and progress.
"""

from __future__ import annotations

import argparse

from swing_lab.web import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the swing_lab web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
