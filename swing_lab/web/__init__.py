"""Phase 3 — local-first web app for team use.

Upload a swing in the browser, get the report, and browse each player's history
and progress. Runs on your machine (or the Jetson); the SQLite library and
on-disk outputs are the same ones the CLI uses. This is the seed that becomes
a multi-user deployment: swap SQLite for Postgres and add auth.
"""

from .app import create_app

__all__ = ["create_app"]
