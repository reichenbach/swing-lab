"""Flask app: upload → analyze → report, plus per-player dashboards.

Processing is synchronous (one coach uploading one clip at a time on a laptop).
For a real multi-user deployment this is where a task queue (RQ/Celery) goes —
flagged, not built, to keep the scaffold honest.
"""

from __future__ import annotations

import secrets
import time
import uuid
from pathlib import Path

from flask import (
    Flask, abort, flash, redirect, render_template,
    request, send_from_directory, url_for,
)

from .. import compare, library
from ..pipeline import analyze_video

ALLOWED = {".mp4", ".mov", ".m4v", ".avi"}


def create_app(db: str | None = None, output_root: str = "output",
               upload_root: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secrets.token_hex(32),      # only signs flash messages; local app
        DB=db or library.DEFAULT_DB,
        OUTPUT_ROOT=output_root,
        UPLOAD_ROOT=upload_root or str(Path(output_root) / "uploads"),
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,  # 512 MB
    )

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            players=library.players(app.config["DB"]),
            swings=library.recent(20, app.config["DB"]),
        )

    @app.route("/upload", methods=["POST"])
    def upload():
        file = request.files.get("video")
        player = (request.form.get("player") or "").strip()
        if not file or file.filename == "":
            flash("Choose a video file.")
            return redirect(url_for("index"))
        if not player:
            flash("Enter a player name.")
            return redirect(url_for("index"))
        if Path(file.filename).suffix.lower() not in ALLOWED:
            flash(f"Unsupported file type. Use {', '.join(sorted(ALLOWED))}.")
            return redirect(url_for("index"))

        Path(app.config["UPLOAD_ROOT"]).mkdir(parents=True, exist_ok=True)
        sid = uuid.uuid4().hex[:8]
        upath = Path(app.config["UPLOAD_ROOT"]) / f"{sid}_{Path(file.filename).name}"
        file.save(str(upath))

        # iPhone slow-mo lies about its frame rate; let the uploader declare the
        # true capture fps so time-based metrics aren't silently wrong.
        try:
            capture_fps = float(request.form.get("capture_fps") or 0) or None
        except ValueError:
            capture_fps = None

        out_dir = Path(app.config["OUTPUT_ROOT"]) / "swings" / sid
        result = analyze_video(str(upath), str(out_dir), player=player,
                               capture_fps=capture_fps)

        prior = library.baseline(player, app.config["DB"])
        scalars = library.scalars_from_metrics(result.metrics)
        row_id = library.record(player, str(upath), scalars,
                                db=app.config["DB"], report=result.report,
                                out_dir=str(out_dir))
        if prior is None:
            flash(f"Saved first baseline for {player}.")
        return redirect(url_for("swing", swing_id=row_id))

    @app.route("/swing/<int:swing_id>")
    def swing(swing_id: int):
        row = library.get(swing_id, app.config["DB"])
        if not row:
            abort(404)
        return render_template("swing.html", row=row, report=row.get("report") or {})

    @app.route("/player/<name>")
    def player(name: str):
        hist = library.history(name, app.config["DB"])
        if not hist:
            abort(404)
        progress = None
        if len(hist) >= 2:
            progress = compare.diff(hist[0]["metrics"], hist[-1]["metrics"])
        return render_template("player.html", name=name, history=hist, progress=progress)

    @app.route("/media/<int:swing_id>/<path:filename>")
    def media(swing_id: int, filename: str):
        row = library.get(swing_id, app.config["DB"])
        if not row or not row.get("out_dir"):
            abort(404)
        # Flask resolves relative paths against the package, not the working
        # directory, and uploads store a relative out_dir.
        return send_from_directory(Path(row["out_dir"]).resolve(), filename)

    @app.template_filter("ts")
    def _ts(value):
        try:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
        except (TypeError, ValueError):
            return value

    return app
