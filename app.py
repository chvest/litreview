import io
import os
import random
import zipfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics import cohen_kappa_score
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "litreview-dev-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///litreview.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

# ── Models ────────────────────────────────────────────────────────────────────

class Project(db.Model):
    __tablename__ = "projects"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Inclusion threshold: "any" | "majority" | "all" | integer string e.g. "2"
    threshold = db.Column(db.String(20), default="any", server_default="any", nullable=False)
    papers = db.relationship("Paper", backref="project", lazy=True, cascade="all, delete-orphan")
    criteria = db.relationship("Criterion", backref="project", lazy=True, cascade="all, delete-orphan")
    reviewers = db.relationship("Reviewer", backref="project", lazy=True, cascade="all, delete-orphan")


class Paper(db.Model):
    __tablename__ = "papers"
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    title = db.Column(db.Text)
    authors = db.Column(db.Text)
    abstract = db.Column(db.Text)
    year = db.Column(db.Integer)
    doi = db.Column(db.String(300))
    source = db.Column(db.String(200))
    reviews = db.relationship("Review", backref="paper", lazy=True, cascade="all, delete-orphan")
    order_entries = db.relationship("ReviewOrder", backref="paper", lazy=True, cascade="all, delete-orphan")


class Criterion(db.Model):
    __tablename__ = "criteria"
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'inclusion' or 'exclusion'
    code = db.Column(db.String(20))
    description = db.Column(db.Text, nullable=False)
    sort_order = db.Column(db.Integer, default=0)


class Reviewer(db.Model):
    __tablename__ = "reviewers"
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    reviews = db.relationship("Review", backref="reviewer", lazy=True, cascade="all, delete-orphan")
    order_entries = db.relationship("ReviewOrder", backref="reviewer", lazy=True, cascade="all, delete-orphan")


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("reviewers.id"), nullable=False)
    stage = db.Column(db.String(20), nullable=False)  # title | abstract | fulltext
    decision = db.Column(db.String(20), nullable=False)  # include | exclude | uncertain
    exclusion_criteria = db.Column(db.String(500))  # comma-separated criterion IDs
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("paper_id", "reviewer_id", "stage"),)


class ReviewOrder(db.Model):
    __tablename__ = "review_orders"
    id = db.Column(db.Integer, primary_key=True)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("reviewers.id"), nullable=False)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    stage = db.Column(db.String(20), nullable=False)
    position = db.Column(db.Integer, nullable=False)
    __table_args__ = (db.UniqueConstraint("reviewer_id", "paper_id", "stage"),)


class PilotBatch(db.Model):
    __tablename__ = "pilot_batches"
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    stage = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default="active")  # active | complete
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    papers = db.relationship("PilotPaper", backref="batch", lazy=True, cascade="all, delete-orphan")


class PilotPaper(db.Model):
    __tablename__ = "pilot_papers"
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("pilot_batches.id"), nullable=False)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    __table_args__ = (db.UniqueConstraint("batch_id", "paper_id"),)


class GroupDecision(db.Model):
    """Manually set consensus override for a paper+stage, independent of reviewer records."""
    __tablename__ = "group_decisions"
    id         = db.Column(db.Integer, primary_key=True)
    paper_id   = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    stage      = db.Column(db.String(20), nullable=False)
    decision   = db.Column(db.String(20), nullable=False)   # include | uncertain | exclude
    notes      = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("paper_id", "stage"),)


# ── Helpers ───────────────────────────────────────────────────────────────────

STAGES = ["title", "abstract", "fulltext"]
STAGE_LABELS = {"title": "Title Review", "abstract": "Abstract Review", "fulltext": "Full-Text Review"}

# Make constants available in all templates
@app.context_processor
def inject_globals():
    return dict(STAGE_LABELS=STAGE_LABELS, STAGES=STAGES)


def get_active_pilot(project_id, stage):
    """Return the active PilotBatch for (project, stage), or None."""
    return PilotBatch.query.filter_by(
        project_id=project_id, stage=stage, status="active"
    ).first()


def get_eligible_papers(project_id, stage, ignore_pilot=False, threshold=None):
    """Papers eligible for review at the given stage.

    title:    all papers
    abstract: papers that passed title review under the project threshold
    fulltext: papers that passed abstract review under the project threshold

    When an active PilotBatch exists (and ignore_pilot is False), only those
    pilot papers are returned so the review flow stays within the pilot.

    threshold overrides the project-level setting when provided explicitly.
    """
    if threshold is None:
        project = Project.query.get(project_id)
        threshold = (project.threshold if project and project.threshold else "any")

    all_papers = Paper.query.filter_by(project_id=project_id).all()
    if stage == "title":
        eligible = all_papers
    else:
        prev_stage = "title" if stage == "abstract" else "abstract"
        eligible = []
        for paper in all_papers:
            prev_reviews = Review.query.filter_by(paper_id=paper.id, stage=prev_stage).all()
            if passes_threshold(prev_reviews, threshold):
                eligible.append(paper)

    if not ignore_pilot:
        pilot = get_active_pilot(project_id, stage)
        if pilot:
            pilot_ids = {pp.paper_id for pp in pilot.papers}
            eligible = [p for p in eligible if p.id in pilot_ids]
    return eligible


def ensure_review_order(reviewer_id, project_id, stage):
    """Create a randomized review order if one doesn't already exist."""
    if ReviewOrder.query.filter_by(reviewer_id=reviewer_id, stage=stage).first():
        return
    papers = get_eligible_papers(project_id, stage)
    ids = [p.id for p in papers]
    random.shuffle(ids)
    for pos, paper_id in enumerate(ids):
        db.session.add(ReviewOrder(reviewer_id=reviewer_id, paper_id=paper_id,
                                   stage=stage, position=pos))
    db.session.commit()


def get_next_paper(reviewer_id, project_id, stage):
    """Return the next unreviewed paper, or None if done."""
    orders = (ReviewOrder.query
              .filter_by(reviewer_id=reviewer_id, stage=stage)
              .order_by(ReviewOrder.position)
              .all())
    reviewed_ids = {r.paper_id for r in
                    Review.query.filter_by(reviewer_id=reviewer_id, stage=stage).all()}
    for o in orders:
        if o.paper_id not in reviewed_ids:
            return o.paper
    return None


def get_prev_paper(reviewer_id, stage, current_paper_id):
    """Return the paper immediately before current_paper_id in the reviewer's order, or None."""
    current = ReviewOrder.query.filter_by(
        reviewer_id=reviewer_id, paper_id=current_paper_id, stage=stage
    ).first()
    if not current or current.position == 0:
        return None
    prev = ReviewOrder.query.filter_by(
        reviewer_id=reviewer_id, stage=stage, position=current.position - 1
    ).first()
    return prev.paper if prev else None


def get_next_paper_in_order(reviewer_id, stage, current_paper_id):
    """Return the paper immediately after current_paper_id in the reviewer's order, or None."""
    current = ReviewOrder.query.filter_by(
        reviewer_id=reviewer_id, paper_id=current_paper_id, stage=stage
    ).first()
    if not current:
        return None
    nxt = ReviewOrder.query.filter_by(
        reviewer_id=reviewer_id, stage=stage, position=current.position + 1
    ).first()
    return nxt.paper if nxt else None


def stage_stats(reviewer_id, stage):
    total = ReviewOrder.query.filter_by(reviewer_id=reviewer_id, stage=stage).count()
    done = Review.query.filter_by(reviewer_id=reviewer_id, stage=stage).count()
    return {"total": total, "done": done, "remaining": total - done}


def interpret_agreement(k):
    if k < 0:
        return "Poor"
    if k < 0.20:
        return "Slight"
    if k < 0.40:
        return "Fair"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Almost Perfect"


def gwets_ac1(y1, y2, n_categories=3):
    """Gwet's AC1 inter-rater agreement coefficient.

    More robust than Cohen's κ when label distributions are highly skewed
    (the 'prevalence problem') — common in systematic reviews where most
    papers are excluded.  Uses average marginal probabilities rather than
    products, so high exclusion rates don't artificially inflate expected
    chance agreement.
    """
    n = len(y1)
    if n == 0:
        return 0.0
    K = n_categories
    p_o = sum(1 for a, b in zip(y1, y2) if a == b) / n
    # Average marginal probability for each category k
    p_e = 0.0
    for k in range(K):
        pi_k = (y1.count(k) + y2.count(k)) / (2 * n)
        p_e += pi_k * (1 - pi_k)
    p_e /= (K - 1)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def normalize_title(t):
    """Lowercase, collapse whitespace — used for fuzzy title matching."""
    return " ".join(str(t).lower().strip().split()) if t else ""


def normalize_doi(d):
    """Strip common prefixes and lowercase for DOI comparison."""
    s = str(d).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def passes_threshold(reviews, threshold):
    """Return True if a paper passes to the next stage under the given threshold.

    reviews   — list of Review objects for one paper at one stage
    threshold — "any"  : at least 1 reviewer did not exclude  (most inclusive)
                "majority": strictly more than half did not exclude
                "all"  : every reviewer must not have excluded (most strict)
                "<int>": at least that many reviewers did not exclude
    """
    if not reviews:
        return False
    non_excluded = sum(1 for r in reviews if r.decision != "exclude")
    total = len(reviews)
    if threshold == "any":
        return non_excluded >= 1
    if threshold == "majority":
        return non_excluded >= (total // 2 + 1)
    if threshold == "all":
        return non_excluded == total
    try:
        return non_excluded >= int(threshold)
    except (ValueError, TypeError):
        return non_excluded >= 1   # safe fallback


def read_upload_file(filepath):
    """Read a CSV or Excel upload into a DataFrame.

    For CSV files uses sep=None / Python engine so that both comma- and
    semicolon-delimited files (common in European Excel exports) are handled
    automatically.  Old-style .xls (BIFF) files require xlrd; .xlsx uses
    openpyxl (bundled with pandas).
    """
    ext = filepath.lower()
    if ext.endswith(".xlsx"):
        return pd.read_excel(filepath, engine="openpyxl")
    if ext.endswith(".xls"):
        return pd.read_excel(filepath, engine="xlrd")
    return pd.read_csv(filepath, encoding="utf-8-sig", sep=None, engine="python")


# Known column-name aliases for common literature databases.
# Keys match the form field names (without the "col_" prefix handled separately).
_COL_ALIASES = {
    "col_title": [
        "title", "document title", "article title", "paper title",
        "ti", "article name",
    ],
    "col_authors": [
        "authors", "author", "author names", "author full names",
        "au", "by",
    ],
    "col_abstract": ["abstract", "ab", "description"],
    "col_year": [
        "year", "publication year", "py", "pub year",
        "publication date",          # fallback if year col absent
    ],
    "col_doi": ["doi", "di", "digital object identifier"],
    "col_source": [
        "source", "source title", "publication title", "journal",
        "journal title", "so", "book series title",
    ],
}


def _write_assignment_xlsx(filepath, papers, project_id):
    """Write an assignment Excel file with a Papers sheet and a Criteria sheet."""
    papers_df = pd.DataFrame([{
        "Title":    p.title    or "",
        "Authors":  p.authors  or "",
        "Abstract": p.abstract or "",
        "Year":     p.year     or "",
        "DOI":      p.doi      or "",
        "Decision": "",
        "Notes":    "",
    } for p in papers])

    criteria = Criterion.query.filter_by(project_id=project_id).order_by(
        Criterion.type, Criterion.sort_order).all()
    criteria_df = pd.DataFrame([{
        "Type":        c.type.capitalize(),
        "Code":        c.code or "",
        "Description": c.description or "",
    } for c in criteria]) if criteria else pd.DataFrame(
        columns=["Type", "Code", "Description"])

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        papers_df.to_excel(writer, sheet_name="Papers", index=False)
        criteria_df.to_excel(writer, sheet_name="Criteria", index=False)


def detect_column_mapping(columns):
    """Return a {field: column_name} dict by matching column headers against
    known aliases for IEEE, Web of Science, Scopus, PubMed, etc."""
    lower_map = {c.strip().lower(): c for c in columns}
    mapping = {}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                mapping[field] = lower_map[alias]
                break
    return mapping


def parse_decision_value(val):
    """Map 0 / 0.5 / 1 (and string variants, including comma decimal) to
    include / uncertain / exclude."""
    try:
        v = float(str(val).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None
    if v < 0.25:
        return "exclude"
    if v < 0.75:
        return "uncertain"
    return "include"


def current_reviewer(pid):
    rid = session.get(f"reviewer_{pid}")
    if not rid:
        return None
    r = Reviewer.query.get(rid)
    if r and r.project_id == pid:
        return r
    session.pop(f"reviewer_{pid}", None)
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    projects = Project.query.order_by(Project.created_at.desc()).all()
    return render_template("index.html", projects=projects)


@app.route("/project/new", methods=["POST"])
def new_project():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    if not name:
        flash("Project name is required.", "danger")
        return redirect(url_for("index"))
    p = Project(name=name, description=description)
    db.session.add(p)
    db.session.commit()
    return redirect(url_for("project_dashboard", pid=p.id))


@app.route("/project/<int:pid>/settings", methods=["GET", "POST"])
def project_settings(pid):
    project = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    n_reviewers = len(reviewers)

    if request.method == "POST":
        project.name = request.form.get("name", "").strip() or project.name
        project.description = request.form.get("description", "").strip() or None

        raw = request.form.get("threshold", "any")
        # "min_n" option: use the companion number field
        if raw == "min_n":
            try:
                n = int(request.form.get("threshold_n", 1))
                raw = str(max(1, n))
            except ValueError:
                raw = "any"
        project.threshold = raw

        db.session.commit()
        flash("Project settings saved.", "success")
        return redirect(url_for("project_settings", pid=pid))

    # Determine which radio to pre-select
    t = project.threshold or "any"
    threshold_mode = t if t in ("any", "majority", "all") else "min_n"
    threshold_n    = int(t) if threshold_mode == "min_n" else 2

    THRESHOLD_LABELS = {
        "any":      "Any — paper advances if at least one reviewer included or was uncertain (most inclusive)",
        "majority": "Majority — more than half of reviewers must have included or been uncertain",
        "all":      "All — every reviewer must have included or been uncertain (most strict)",
        "min_n":    "Minimum N — at least N reviewers must have included or been uncertain",
    }

    return render_template("project_settings.html", project=project,
                           n_reviewers=n_reviewers,
                           threshold_mode=threshold_mode,
                           threshold_n=threshold_n,
                           THRESHOLD_LABELS=THRESHOLD_LABELS)


@app.route("/project/<int:pid>/delete-papers", methods=["POST"])
def delete_all_papers(pid):
    """Delete every paper (and all dependent rows) for this project."""
    Project.query.get_or_404(pid)
    paper_ids = [p.id for p in Paper.query.filter_by(project_id=pid).all()]
    if paper_ids:
        ReviewOrder.query.filter(ReviewOrder.paper_id.in_(paper_ids)).delete(synchronize_session=False)
        PilotPaper.query.filter(PilotPaper.paper_id.in_(paper_ids)).delete(synchronize_session=False)
        Review.query.filter(Review.paper_id.in_(paper_ids)).delete(synchronize_session=False)
        Paper.query.filter(Paper.id.in_(paper_ids)).delete(synchronize_session=False)
        db.session.commit()
    flash(f"All {len(paper_ids)} papers and their decisions have been removed.", "info")
    return redirect(url_for("project_settings", pid=pid))


@app.route("/project/<int:pid>/review/<stage>/reset", methods=["POST"])
def reset_stage(pid, stage):
    """Delete all review decisions for one stage."""
    Project.query.get_or_404(pid)
    if stage not in STAGES:
        return redirect(url_for("project_dashboard", pid=pid))
    paper_ids = [p.id for p in Paper.query.filter_by(project_id=pid).all()]
    if paper_ids:
        ReviewOrder.query.filter(
            ReviewOrder.paper_id.in_(paper_ids),
            ReviewOrder.stage == stage
        ).delete(synchronize_session=False)
        count = Review.query.filter(
            Review.paper_id.in_(paper_ids),
            Review.stage == stage
        ).delete(synchronize_session=False)
        db.session.commit()
    else:
        count = 0
    label = STAGE_LABELS.get(stage, stage)
    flash(f"{label} reset — {count} decision(s) removed.", "info")
    return redirect(url_for("project_dashboard", pid=pid))


@app.route("/project/<int:pid>")
def project_home(pid):
    """Landing page — choose Lead Reviewer dashboard or Assignment Workspace."""
    project = Project.query.get_or_404(pid)
    total   = Paper.query.filter_by(project_id=pid).count()
    return render_template("project_home.html", project=project, total=total)


@app.route("/project/<int:pid>/dashboard")
def project_dashboard(pid):
    project = Project.query.get_or_404(pid)
    total = Paper.query.filter_by(project_id=pid).count()

    stages_info = {}
    all_reviewers = Reviewer.query.filter_by(project_id=pid).all()
    for stage in STAGES:
        eligible = len(get_eligible_papers(pid, stage, ignore_pilot=True))
        reviewed_papers = (db.session.query(Review.paper_id)
                           .join(Paper)
                           .filter(Paper.project_id == pid, Review.stage == stage)
                           .distinct().count())
        active_pilot = get_active_pilot(pid, stage)
        pilot_size = len(active_pilot.papers) if active_pilot else 0

        # Per-reviewer pilot completion
        pilot_reviewer_progress = []
        if active_pilot and all_reviewers:
            pilot_paper_ids = [pp.paper_id for pp in active_pilot.papers]
            for r in all_reviewers:
                done = (Review.query
                        .filter_by(reviewer_id=r.id, stage=stage)
                        .filter(Review.paper_id.in_(pilot_paper_ids))
                        .count())
                pilot_reviewer_progress.append({
                    "name": r.name, "done": done, "total": pilot_size,
                    "complete": done >= pilot_size
                })

        stages_info[stage] = {"eligible": eligible, "reviewed": reviewed_papers,
                               "label": STAGE_LABELS[stage], "pilot_size": pilot_size,
                               "pilot_reviewer_progress": pilot_reviewer_progress}

    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    criteria_count = Criterion.query.filter_by(project_id=pid).count()
    return render_template("project.html", project=project, total=total,
                           stages_info=stages_info, reviewers=reviewers,
                           criteria_count=criteria_count)


# ── Import ────────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/import", methods=["GET", "POST"])
def import_papers(pid):
    project = Project.query.get_or_404(pid)
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)
        try:
            df = read_upload_file(filepath)
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(request.url)

        # Store only the filepath — columns and preview are re-read from disk
        # on the next step to avoid overflowing the 4KB session cookie limit.
        session["import_file"] = filepath
        return redirect(url_for("import_columns", pid=pid))
    return render_template("import.html", project=project)


@app.route("/project/<int:pid>/import/columns", methods=["GET", "POST"])
def import_columns(pid):
    project = Project.query.get_or_404(pid)
    if "import_file" not in session:
        return redirect(url_for("import_papers", pid=pid))

    filepath = session["import_file"]
    try:
        df_preview = read_upload_file(filepath)
    except Exception as e:
        flash(f"Error reading file: {e}", "danger")
        return redirect(url_for("import_papers", pid=pid))
    columns  = list(df_preview.columns)
    preview  = df_preview.head(3).fillna("").astype(str).to_dict("records")
    suggested = detect_column_mapping(columns)

    if request.method == "POST":
        mapping = {
            "title":    request.form.get("col_title"),
            "authors":  request.form.get("col_authors"),
            "abstract": request.form.get("col_abstract"),
            "year":     request.form.get("col_year"),
            "doi":      request.form.get("col_doi"),
            "source":   request.form.get("col_source"),
        }

        filepath = session["import_file"]
        try:
            df = read_upload_file(filepath)
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(url_for("import_papers", pid=pid))

        def col(key):
            c = mapping.get(key)
            if not c:
                return ""
            val = row.get(c) if hasattr(row, 'get') else ""
            if pd.isna(val):
                return ""
            s = str(val).strip()
            return "" if s.lower() == "nan" else s

        # Build normalised lookup sets for fast duplicate detection
        existing_dois   = {normalize_doi(p.doi)    for p in Paper.query.filter_by(project_id=pid).all() if p.doi}
        existing_titles = {normalize_title(p.title) for p in Paper.query.filter_by(project_id=pid).all() if p.title}

        imported = skipped = no_doi = 0
        for _, row in df.iterrows():
            title = col("title")
            doi   = col("doi")
            if not title and not doi:
                skipped += 1
                continue

            ndoi   = normalize_doi(doi)     if doi   else ""
            ntitle = normalize_title(title) if title else ""

            # Duplicate check (normalised)
            if (ndoi and ndoi in existing_dois) or (ntitle and ntitle in existing_titles):
                skipped += 1
                continue

            year = None
            raw_year = col("year")
            if raw_year:
                try:
                    year = int(float(raw_year))
                except ValueError:
                    pass

            db.session.add(Paper(project_id=pid, title=title or None,
                                 authors=col("authors") or None,
                                 abstract=col("abstract") or None,
                                 year=year,
                                 doi=doi or None,
                                 source=col("source") or None))
            if ndoi:
                existing_dois.add(ndoi)
            if ntitle:
                existing_titles.add(ntitle)
            if not doi:
                no_doi += 1
            imported += 1

        db.session.commit()
        for key in ("import_file", "import_columns", "import_preview"):
            session.pop(key, None)

        msg = f"Imported {imported} paper(s). Skipped {skipped} (duplicates or empty rows)."
        if no_doi:
            msg += f" {no_doi} paper(s) have no DOI — they are included but flagged."
        flash(msg, "success")
        return redirect(url_for("project_dashboard", pid=pid))

    return render_template("column_map.html", project=project,
                           columns=columns, preview=preview,
                           suggested=suggested)


# ── Import assignment (papers from a LitReview assignment file) ───────────────

@app.route("/project/<int:pid>/import-assignment", methods=["GET", "POST"])
def import_assignment_papers(pid):
    """Import papers from a LitReview-generated assignment CSV/Excel.

    The assignment format has known column names (Title, Authors, Abstract,
    Year, DOI, Decision, Notes), so no column-mapping step is needed.
    Duplicate papers are skipped by DOI or normalised title.
    """
    project = Project.query.get_or_404(pid)

    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        try:
            df = read_upload_file(filepath)
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(request.url)

        # Import criteria from a "Criteria" sheet if present (xlsx only)
        criteria_imported = 0
        if filepath.lower().endswith((".xlsx", ".xls")):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(filepath, read_only=True)
                if "Criteria" in wb.sheetnames:
                    cdf = pd.read_excel(filepath, sheet_name="Criteria", engine="openpyxl")
                    c_col = {c.strip().lower(): c for c in cdf.columns}
                    for _, crow in cdf.iterrows():
                        ctype = str(crow.get(c_col.get("type", ""), "") or "").strip().lower()
                        ccode = str(crow.get(c_col.get("code", ""), "") or "").strip()
                        cdesc = str(crow.get(c_col.get("description", ""), "") or "").strip()
                        if ctype not in ("inclusion", "exclusion") or not cdesc:
                            continue
                        # Skip if identical criterion already exists
                        exists = Criterion.query.filter_by(
                            project_id=pid, type=ctype, description=cdesc).first()
                        if not exists:
                            db.session.add(Criterion(
                                project_id=pid, type=ctype,
                                code=ccode or None, description=cdesc))
                            criteria_imported += 1
            except Exception:
                pass  # criteria import is best-effort

        # Case-insensitive column lookup
        col = {c.strip().lower(): c for c in df.columns}

        def get(row, name, default=""):
            key = col.get(name.lower())
            if key is None:
                return default
            v = row[key]
            return "" if (v != v or v is None) else str(v).strip()  # handles NaN

        if "title" not in col and "doi" not in col:
            flash("File must have at least a Title or DOI column. "
                  "Is this a LitReview assignment file?", "danger")
            return redirect(request.url)

        # Build duplicate-check sets from existing papers
        existing_dois   = {normalize_doi(p.doi)   for p in Paper.query.filter_by(project_id=pid).all() if p.doi}
        existing_titles = {normalize_title(p.title) for p in Paper.query.filter_by(project_id=pid).all() if p.title}

        imported = skipped = 0
        for _, row in df.iterrows():
            title    = get(row, "title")
            doi      = get(row, "doi")
            authors  = get(row, "authors")
            abstract = get(row, "abstract")
            year_raw = get(row, "year")

            ndoi   = normalize_doi(doi)     if doi   else ""
            ntitle = normalize_title(title) if title else ""

            if (ndoi and ndoi in existing_dois) or (ntitle and ntitle in existing_titles):
                skipped += 1
                continue

            try:
                year = int(float(year_raw)) if year_raw else None
            except (ValueError, TypeError):
                year = None

            db.session.add(Paper(
                project_id=pid,
                title    = title    or None,
                authors  = authors  or None,
                abstract = abstract or None,
                year     = year,
                doi      = doi      or None,
            ))
            if ndoi:
                existing_dois.add(ndoi)
            if ntitle:
                existing_titles.add(ntitle)
            imported += 1

        db.session.commit()
        msg = f"Imported {imported} paper(s)."
        if skipped:
            msg += f" Skipped {skipped} duplicate(s)."
        if criteria_imported:
            msg += f" Imported {criteria_imported} review criteria from the assignment."
        flash(msg, "success")
        return redirect(url_for("assignment_workspace", pid=pid))

    return render_template("import_assignment.html", project=project)


# ── Assignment workspace ──────────────────────────────────────────────────────

@app.route("/project/<int:pid>/workspace", methods=["GET", "POST"])
def assignment_workspace(pid):
    """Simplified view for a reviewer working through an assignment.

    Shows only what an assignment reviewer needs: their name, progress per
    stage, review criteria, and an export button.
    """
    project   = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    reviewer  = current_reviewer(pid)

    if request.method == "POST":
        action = request.form.get("action")
        if action == "set_reviewer":
            rid = request.form.get("reviewer_id", type=int)
            new_name = request.form.get("new_reviewer_name", "").strip()
            if rid:
                r = Reviewer.query.get(rid)
                if r and r.project_id == pid:
                    session[f"reviewer_{pid}"] = r.id
            elif new_name:
                r = Reviewer.query.filter_by(project_id=pid, name=new_name).first()
                if not r:
                    r = Reviewer(project_id=pid, name=new_name)
                    db.session.add(r)
                    db.session.commit()
                session[f"reviewer_{pid}"] = r.id
            return redirect(url_for("assignment_workspace", pid=pid))

        if action == "clear_assignment":
            paper_ids = [p.id for p in Paper.query.filter_by(project_id=pid).all()]
            if paper_ids:
                ReviewOrder.query.filter(ReviewOrder.paper_id.in_(paper_ids)).delete(synchronize_session=False)
                PilotPaper.query.filter(PilotPaper.paper_id.in_(paper_ids)).delete(synchronize_session=False)
                Review.query.filter(Review.paper_id.in_(paper_ids)).delete(synchronize_session=False)
                Paper.query.filter(Paper.id.in_(paper_ids)).delete(synchronize_session=False)
                db.session.commit()
            flash("All imported papers and decisions have been removed.", "info")
            return redirect(url_for("assignment_workspace", pid=pid))

    inclusion_criteria = Criterion.query.filter_by(
        project_id=pid, type="inclusion").order_by(Criterion.sort_order).all()
    exclusion_criteria = Criterion.query.filter_by(
        project_id=pid, type="exclusion").order_by(Criterion.sort_order).all()

    stage_info = []
    for stage in STAGES:
        papers = get_eligible_papers(pid, stage, ignore_pilot=True)
        if not papers:
            continue
        if reviewer:
            revs = {r.paper_id for r in
                    Review.query.filter_by(reviewer_id=reviewer.id, stage=stage)
                    .join(Paper).filter(Paper.project_id == pid).all()}
            done = len(revs)
        else:
            done = 0
        stage_info.append({
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "total": len(papers),
            "done":  done,
            "pct":   int(done / len(papers) * 100) if papers else 0,
        })

    return render_template("assignment_workspace.html",
                           project=project,
                           reviewer=reviewer,
                           reviewers=reviewers,
                           stage_info=stage_info,
                           inclusion_criteria=inclusion_criteria,
                           exclusion_criteria=exclusion_criteria,
                           DECISION_TO_NUM=DECISION_TO_NUM)


# ── Import reviews ────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/import-reviews", methods=["GET", "POST"])
def import_reviews_upload(pid):
    project = Project.query.get_or_404(pid)
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "rev_" + filename)
        f.save(filepath)
        try:
            df = read_upload_file(filepath)
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(request.url)

        # Only keep the filepath in the session — columns/preview/samples are
        # derived from the file itself and can exceed the 4 KB cookie limit.
        session["rev_import_file"] = filepath

        return redirect(url_for("import_reviews_map", pid=pid))
    return render_template("import_reviews.html", project=project)


@app.route("/project/<int:pid>/import-reviews/map", methods=["GET", "POST"])
def import_reviews_map(pid):
    project = Project.query.get_or_404(pid)
    if "rev_import_file" not in session:
        return redirect(url_for("import_reviews_upload", pid=pid))

    # Re-derive display data from the saved file (avoids 4 KB cookie-session limit)
    filepath = session["rev_import_file"]
    try:
        _df = read_upload_file(filepath)
    except Exception as e:
        flash(f"Could not re-read upload file: {e}", "danger")
        session.pop("rev_import_file", None)
        return redirect(url_for("import_reviews_upload", pid=pid))

    columns = list(_df.columns)
    preview = _df.head(3).fillna("").astype(str).to_dict("records")
    value_samples = {}
    for col in _df.columns:
        unique = _df[col].dropna().unique()[:6]
        value_samples[col] = [str(v) for v in unique]

    reviewers = Reviewer.query.filter_by(project_id=pid).all()

    if request.method == "POST":
        col_title    = request.form.get("col_title")
        col_decision = request.form.get("col_decision")
        col_doi      = request.form.get("col_doi") or None
        stage        = request.form.get("stage", "title")
        overwrite    = request.form.get("overwrite") == "yes"

        # Resolve / create reviewer
        reviewer_mode = request.form.get("reviewer_mode", "existing")
        if reviewer_mode == "new":
            new_name = request.form.get("new_reviewer_name", "").strip()
            if not new_name:
                flash("Enter a name for the new reviewer.", "danger")
                return redirect(request.url)
            reviewer = Reviewer.query.filter_by(project_id=pid, name=new_name).first()
            if not reviewer:
                reviewer = Reviewer(project_id=pid, name=new_name)
                db.session.add(reviewer)
                db.session.commit()
        else:
            rid = request.form.get("existing_reviewer_id", type=int)
            reviewer = Reviewer.query.get(rid) if rid else None
            if not reviewer or reviewer.project_id != pid:
                flash("Select a valid reviewer.", "danger")
                return redirect(request.url)

        if not col_title or not col_decision:
            flash("Title and Decision columns are required.", "danger")
            return redirect(request.url)

        filepath = session["rev_import_file"]
        try:
            df = read_upload_file(filepath)
        except Exception as e:
            flash(f"Error reading file: {e}", "danger")
            return redirect(url_for("import_reviews_upload", pid=pid))

        # Build lookup dicts for existing papers
        papers = Paper.query.filter_by(project_id=pid).all()
        title_map = {normalize_title(p.title): p for p in papers if p.title}
        doi_map   = {normalize_doi(p.doi): p   for p in papers if p.doi}

        imported = skipped = unmatched = bad_value = 0

        for _, row in df.iterrows():
            raw_title    = str(row.get(col_title, "") or "").strip()
            raw_decision = row.get(col_decision)
            raw_doi      = str(row.get(col_doi, "") or "").strip() if col_doi else ""

            decision = parse_decision_value(raw_decision)
            if decision is None:
                bad_value += 1
                continue

            # Match paper: DOI first, then title
            paper = None
            if raw_doi:
                paper = doi_map.get(normalize_doi(raw_doi))
            if paper is None and raw_title:
                paper = title_map.get(normalize_title(raw_title))
            if paper is None:
                unmatched += 1
                continue

            existing = Review.query.filter_by(
                paper_id=paper.id, reviewer_id=reviewer.id, stage=stage
            ).first()

            if existing:
                if overwrite:
                    existing.decision = decision
                    existing.notes = f"[imported]"
                    imported += 1
                else:
                    skipped += 1
            else:
                db.session.add(Review(
                    paper_id=paper.id, reviewer_id=reviewer.id,
                    stage=stage, decision=decision, notes="[imported]"
                ))
                imported += 1

        db.session.commit()

        # Create review order for this reviewer so they appear correctly on the
        # review start page (ensure_review_order skips if order already exists)
        ensure_review_order(reviewer.id, pid, stage)

        session.pop("rev_import_file", None)

        parts = [f"Imported {imported} review decision(s)"]
        if skipped:    parts.append(f"{skipped} skipped (already reviewed)")
        if unmatched:  parts.append(f"{unmatched} rows had no matching paper")
        if bad_value:  parts.append(f"{bad_value} rows had an unrecognised decision value")
        flash(". ".join(parts) + ".", "success" if imported else "warning")

        # Warn if the reviewer is missing decisions for any expected papers.
        # When a pilot is active, the expected set is just the pilot batch;
        # otherwise it's all eligible papers for the stage.
        active_pilot = get_active_pilot(pid, stage)
        if active_pilot:
            expected_ids = {pp.paper_id for pp in active_pilot.papers}
            context = "pilot"
        else:
            expected_ids = {p.id for p in get_eligible_papers(pid, stage, ignore_pilot=True)}
            context = stage
        reviewed_ids = {r.paper_id for r in
                        Review.query.filter_by(reviewer_id=reviewer.id, stage=stage)
                        .join(Paper).filter(Paper.project_id == pid).all()}
        missing = len(expected_ids - reviewed_ids)
        if missing:
            flash(
                f"⚠ {reviewer.name} is still missing decisions for {missing} "
                f"{context} paper(s). You can import again or review them manually.",
                "warning"
            )

        return redirect(url_for("project_dashboard", pid=pid))

    active_pilots = {s: get_active_pilot(pid, s) for s in STAGES}

    return render_template("import_reviews_map.html", project=project,
                           columns=columns, preview=preview,
                           value_samples=value_samples,
                           reviewers=reviewers,
                           stages=STAGES, stage_labels=STAGE_LABELS,
                           active_pilots=active_pilots)


# ── Paper list ────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/papers")
def paper_list(pid):
    project = Project.query.get_or_404(pid)
    papers  = Paper.query.filter_by(project_id=pid).order_by(Paper.id).all()
    return render_template("papers.html", project=project, papers=papers,
                           stages=STAGES, stage_labels=STAGE_LABELS)


@app.route("/project/<int:pid>/papers/<int:paper_id>/delete", methods=["POST"])
def delete_paper(pid, paper_id):
    paper = Paper.query.get_or_404(paper_id)
    if paper.project_id != pid:
        return redirect(url_for("paper_list", pid=pid))
    db.session.delete(paper)
    db.session.commit()
    flash("Paper deleted.", "success")
    return redirect(url_for("paper_list", pid=pid))


# ── Criteria ──────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/criteria", methods=["GET", "POST"])
def manage_criteria(pid):
    project = Project.query.get_or_404(pid)
    if request.method == "POST":
        ctype = request.form.get("type", "inclusion")
        code = request.form.get("code", "").strip()
        description = request.form.get("description", "").strip()
        if not description:
            flash("Description is required.", "danger")
        else:
            if not code:
                prefix = "IC" if ctype == "inclusion" else "EC"
                n = Criterion.query.filter_by(project_id=pid, type=ctype).count()
                code = f"{prefix}{n + 1}"
            db.session.add(Criterion(project_id=pid, type=ctype,
                                     code=code, description=description))
            db.session.commit()
            flash("Criterion added.", "success")

    inclusion = (Criterion.query.filter_by(project_id=pid, type="inclusion")
                 .order_by(Criterion.sort_order, Criterion.id).all())
    exclusion = (Criterion.query.filter_by(project_id=pid, type="exclusion")
                 .order_by(Criterion.sort_order, Criterion.id).all())
    return render_template("criteria.html", project=project,
                           inclusion=inclusion, exclusion=exclusion)


@app.route("/project/<int:pid>/criteria/<int:cid>/delete", methods=["POST"])
def delete_criterion(pid, cid):
    c = Criterion.query.get_or_404(cid)
    db.session.delete(c)
    db.session.commit()
    flash("Criterion deleted.", "success")
    return redirect(url_for("manage_criteria", pid=pid))


# ── Reviewers ─────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/reviewers", methods=["GET", "POST"])
def manage_reviewers(pid):
    project = Project.query.get_or_404(pid)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Name is required.", "danger")
        elif Reviewer.query.filter_by(project_id=pid, name=name).first():
            flash("A reviewer with that name already exists.", "danger")
        else:
            db.session.add(Reviewer(project_id=pid, name=name))
            db.session.commit()
            flash(f'Reviewer "{name}" added.', "success")

    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    reviewer_stats = []
    for r in reviewers:
        stats = {stage: Review.query.filter_by(reviewer_id=r.id, stage=stage).count()
                 for stage in STAGES}
        reviewer_stats.append((r, stats))
    return render_template("reviewers.html", project=project, reviewer_stats=reviewer_stats)


@app.route("/project/<int:pid>/reviewers/<int:rid>/delete", methods=["POST"])
def delete_reviewer(pid, rid):
    r = Reviewer.query.get_or_404(rid)
    db.session.delete(r)
    db.session.commit()
    flash("Reviewer removed.", "success")
    return redirect(url_for("manage_reviewers", pid=pid))


# ── Review flow ───────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/review/<stage>")
def review_start(pid, stage):
    if stage not in STAGES:
        return redirect(url_for("project_dashboard", pid=pid))
    project = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    reviewer = current_reviewer(pid)

    stats = None
    paper_list = []
    if reviewer:
        ensure_review_order(reviewer.id, pid, stage)
        stats = stage_stats(reviewer.id, stage)
        # Build ordered list with decision if any
        orders = (ReviewOrder.query
                  .filter_by(reviewer_id=reviewer.id, stage=stage)
                  .order_by(ReviewOrder.position).all())
        done_map = {r.paper_id: r for r in
                    Review.query.filter_by(reviewer_id=reviewer.id, stage=stage).all()}
        paper_list = [(o.position + 1, o.paper, done_map.get(o.paper_id)) for o in orders]

    active_pilot    = get_active_pilot(pid, stage)
    eligible_count  = len(get_eligible_papers(pid, stage, ignore_pilot=True))
    pilot_size      = len(active_pilot.papers) if active_pilot else 0
    return render_template("review_start.html", project=project,
                           reviewers=reviewers, reviewer=reviewer,
                           stage=stage, stage_label=STAGE_LABELS[stage],
                           stats=stats, eligible_count=eligible_count,
                           pilot_size=pilot_size,
                           paper_list=paper_list, active_pilot=active_pilot)


@app.route("/project/<int:pid>/review/<stage>/set-reviewer", methods=["POST"])
def set_reviewer(pid, stage):
    rid = request.form.get("reviewer_id", type=int)
    r = Reviewer.query.get_or_404(rid)
    if r.project_id != pid:
        flash("Invalid reviewer.", "danger")
    else:
        session[f"reviewer_{pid}"] = rid
    return redirect(url_for("review_start", pid=pid, stage=stage))


@app.route("/project/<int:pid>/review/<stage>/next")
def review_next(pid, stage):
    reviewer = current_reviewer(pid)
    if not reviewer:
        return redirect(url_for("review_start", pid=pid, stage=stage))
    ensure_review_order(reviewer.id, pid, stage)
    paper = get_next_paper(reviewer.id, pid, stage)
    if paper is None:
        return redirect(url_for("review_start", pid=pid, stage=stage))
    return redirect(url_for("review_paper", pid=pid, stage=stage, paper_id=paper.id))


@app.route("/project/<int:pid>/review/<stage>/paper/<int:paper_id>", methods=["GET", "POST"])
def review_paper(pid, stage, paper_id):
    if stage not in STAGES:
        return redirect(url_for("project_dashboard", pid=pid))
    project = Project.query.get_or_404(pid)
    paper = Paper.query.get_or_404(paper_id)
    reviewer = current_reviewer(pid)
    if not reviewer:
        return redirect(url_for("review_start", pid=pid, stage=stage))

    existing = Review.query.filter_by(paper_id=paper_id,
                                      reviewer_id=reviewer.id,
                                      stage=stage).first()

    if request.method == "POST":
        decision = request.form.get("decision")
        notes = request.form.get("notes", "").strip()
        exc_criteria = ",".join(request.form.getlist("exclusion_criteria"))
        if decision not in ("include", "exclude", "uncertain"):
            flash("Invalid decision.", "danger")
        else:
            if existing:
                existing.decision = decision
                existing.notes = notes
                existing.exclusion_criteria = exc_criteria
            else:
                db.session.add(Review(paper_id=paper_id, reviewer_id=reviewer.id,
                                      stage=stage, decision=decision,
                                      notes=notes, exclusion_criteria=exc_criteria))
            db.session.commit()
            next_p = get_next_paper(reviewer.id, pid, stage)
            if next_p:
                return redirect(url_for("review_paper", pid=pid, stage=stage, paper_id=next_p.id))
            # All done — return to this paper; template shows the completion banner
            return redirect(url_for("review_paper", pid=pid, stage=stage, paper_id=paper_id))

    inclusion_criteria = (Criterion.query.filter_by(project_id=pid, type="inclusion")
                          .order_by(Criterion.sort_order, Criterion.id).all())
    exclusion_criteria = (Criterion.query.filter_by(project_id=pid, type="exclusion")
                          .order_by(Criterion.sort_order, Criterion.id).all())
    stats = stage_stats(reviewer.id, stage)
    order_entry = ReviewOrder.query.filter_by(reviewer_id=reviewer.id,
                                              paper_id=paper_id, stage=stage).first()
    position = (order_entry.position + 1) if order_entry else "?"

    # Pre-fill exclusion criteria from existing review
    selected_exc = set()
    if existing and existing.exclusion_criteria:
        selected_exc = set(existing.exclusion_criteria.split(","))

    prev_paper = get_prev_paper(reviewer.id, stage, paper_id)
    prev_paper_id = prev_paper.id if prev_paper else None
    next_paper = get_next_paper_in_order(reviewer.id, stage, paper_id)
    next_paper_id = next_paper.id if next_paper else None
    active_pilot = get_active_pilot(pid, stage)
    next_stage = STAGES[STAGES.index(stage) + 1] if stage != STAGES[-1] else None

    return render_template("review_paper.html", project=project, paper=paper,
                           reviewer=reviewer, stage=stage,
                           stage_label=STAGE_LABELS[stage],
                           stats=stats, existing=existing,
                           inclusion_criteria=inclusion_criteria,
                           exclusion_criteria=exclusion_criteria,
                           selected_exc=selected_exc,
                           position=position,
                           prev_paper_id=prev_paper_id,
                           next_paper_id=next_paper_id,
                           active_pilot=active_pilot,
                           next_stage=next_stage)


@app.route("/project/<int:pid>/review/<stage>/complete")
def review_complete(pid, stage):
    project = Project.query.get_or_404(pid)
    reviewer = current_reviewer(pid)
    stats = stage_stats(reviewer.id, stage) if reviewer else None
    decisions = {}
    if reviewer:
        for d in ("include", "exclude", "uncertain"):
            decisions[d] = (Review.query
                            .filter_by(reviewer_id=reviewer.id, stage=stage, decision=d)
                            .join(Paper).filter(Paper.project_id == pid).count())
    next_stage = STAGES[STAGES.index(stage) + 1] if stage != "fulltext" else None
    return render_template("review_complete.html", project=project,
                           reviewer=reviewer, stage=stage,
                           stage_label=STAGE_LABELS[stage],
                           stats=stats, decisions=decisions,
                           next_stage=next_stage)


# ── Pilot ─────────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/pilot")
def pilot_overview(pid):
    project = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()

    stage_data = {}
    for stage in STAGES:
        active = PilotBatch.query.filter_by(project_id=pid, stage=stage, status="active").first()
        completed_count = PilotBatch.query.filter_by(project_id=pid, stage=stage, status="complete").count()

        reviewer_progress = []
        batch_size = 0
        if active:
            pilot_paper_ids = [pp.paper_id for pp in active.papers]
            batch_size = len(pilot_paper_ids)
            for r in reviewers:
                done = Review.query.filter(
                    Review.reviewer_id == r.id,
                    Review.stage == stage,
                    Review.paper_id.in_(pilot_paper_ids)
                ).count() if pilot_paper_ids else 0
                reviewer_progress.append({
                    "reviewer": r,
                    "done": done,
                    "total": batch_size,
                    "pct": int(done / batch_size * 100) if batch_size else 0,
                })

        stage_data[stage] = {
            "active": active,
            "batch_size": batch_size,
            "completed_count": completed_count,
            "reviewer_progress": reviewer_progress,
            "eligible_count": len(get_eligible_papers(pid, stage, ignore_pilot=True)),
        }

    return render_template("pilot.html", project=project, stage_data=stage_data,
                           stages=STAGES, stage_labels=STAGE_LABELS)


@app.route("/project/<int:pid>/pilot/generate", methods=["POST"])
def pilot_generate(pid):
    stage = request.form.get("stage", "title")
    pct = max(1, min(100, request.form.get("pct", 10, type=int)))

    # Remove any existing active batch for this stage
    existing = PilotBatch.query.filter_by(project_id=pid, stage=stage, status="active").first()
    if existing:
        db.session.delete(existing)
        db.session.commit()

    # Sample from ALL eligible papers (ignore_pilot=True since we just deleted the batch)
    eligible = get_eligible_papers(pid, stage, ignore_pilot=True)
    if not eligible:
        flash("No eligible papers for this stage.", "warning")
        return redirect(url_for("pilot_overview", pid=pid))

    n = max(1, round(len(eligible) * pct / 100))
    sample = random.sample(eligible, min(n, len(eligible)))

    batch = PilotBatch(project_id=pid, stage=stage)
    db.session.add(batch)
    db.session.flush()

    for paper in sample:
        db.session.add(PilotPaper(batch_id=batch.id, paper_id=paper.id))

    # Reset review orders for this stage so they regenerate with the pilot papers
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    for r in reviewers:
        ReviewOrder.query.filter_by(reviewer_id=r.id, stage=stage).delete()

    db.session.commit()
    flash(f"Pilot batch created: {len(sample)} papers ({pct}%) for {STAGE_LABELS[stage]}.", "success")
    return redirect(url_for("pilot_overview", pid=pid))


@app.route("/project/<int:pid>/pilot/reset", methods=["POST"])
def pilot_reset(pid):
    """Delete all pilot reviews and reset review orders so reviewers start the pilot over."""
    stage = request.form.get("stage")
    pilot = PilotBatch.query.filter_by(project_id=pid, stage=stage, status="active").first()
    if not pilot:
        flash("No active pilot for that stage.", "warning")
        return redirect(url_for("pilot_overview", pid=pid))

    pilot_ids = [pp.paper_id for pp in pilot.papers]
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    for r in reviewers:
        for paper_id in pilot_ids:
            Review.query.filter_by(reviewer_id=r.id, paper_id=paper_id, stage=stage).delete()
        ReviewOrder.query.filter_by(reviewer_id=r.id, stage=stage).delete()

    db.session.commit()
    flash("Pilot reviews reset. Reviewers can now redo the pilot batch with refined criteria.", "success")
    return redirect(url_for("pilot_overview", pid=pid))


@app.route("/project/<int:pid>/pilot/export-assignment")
def pilot_export_assignment(pid):
    """Export the active pilot batch as a blank assignment CSV for other reviewers."""
    project = Project.query.get_or_404(pid)
    stage   = request.args.get("stage", "title")

    active = get_active_pilot(pid, stage)
    if not active:
        flash("No active pilot batch for this stage.", "warning")
        return redirect(url_for("pilot_overview", pid=pid))

    paper_ids = [pp.paper_id for pp in active.papers]
    papers = Paper.query.filter(Paper.id.in_(paper_ids)).all()
    project_safe = project.name.replace(" ", "_")
    out = os.path.join(app.config["UPLOAD_FOLDER"],
                       f"pilot_assignment_{pid}_{stage}.xlsx")
    _write_assignment_xlsx(out, papers, pid)
    return send_file(out, as_attachment=True,
                     download_name=f"{project_safe}_{stage}_pilot_assignment.xlsx")


@app.route("/project/<int:pid>/pilot/advance", methods=["POST"])
def pilot_advance(pid):
    """Mark the pilot complete and open the stage for full review."""
    stage = request.form.get("stage")
    keep_reviews = request.form.get("keep_reviews", "yes") == "yes"

    pilot = PilotBatch.query.filter_by(project_id=pid, stage=stage, status="active").first()
    if not pilot:
        flash("No active pilot for that stage.", "warning")
        return redirect(url_for("pilot_overview", pid=pid))

    if not keep_reviews:
        pilot_ids = [pp.paper_id for pp in pilot.papers]
        for r in Reviewer.query.filter_by(project_id=pid).all():
            for paper_id in pilot_ids:
                Review.query.filter_by(reviewer_id=r.id, paper_id=paper_id, stage=stage).delete()

    pilot.status = "complete"

    # Clear review orders so they regenerate with the full eligible paper set
    for r in Reviewer.query.filter_by(project_id=pid).all():
        ReviewOrder.query.filter_by(reviewer_id=r.id, stage=stage).delete()

    db.session.commit()
    action = "kept — pilot papers are already marked as reviewed" if keep_reviews else "discarded"
    flash(f"Pilot complete. Reviews {action}. Full {STAGE_LABELS[stage]} is now active.", "success")
    return redirect(url_for("review_start", pid=pid, stage=stage))


# ── Statistics ────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/statistics")
def statistics(pid):
    project = Project.query.get_or_404(pid)
    total = Paper.query.filter_by(project_id=pid).count()

    funnel = [{"label": "Imported", "count": total}]
    for stage in STAGES:
        eligible = get_eligible_papers(pid, stage, threshold=project.threshold)
        passed = sum(
            1 for p in eligible
            if passes_threshold(
                Review.query.filter_by(paper_id=p.id, stage=stage).all(),
                project.threshold
            )
        )
        funnel.append({
            "label": STAGE_LABELS[stage],
            "eligible": len(eligible),
            "passed": passed,
        })

    # Papers per year (based on furthest-reviewed stage with data)
    stages_done = [s for s in STAGES
                   if Review.query.join(Paper).filter(
                       Paper.project_id == pid, Review.stage == s).count() > 0]

    papers_by_year = defaultdict(int)
    final_included = 0
    if stages_done:
        last = stages_done[-1]
        for paper in Paper.query.filter_by(project_id=pid).all():
            reviews = Review.query.filter_by(paper_id=paper.id, stage=last).all()
            if passes_threshold(reviews, project.threshold):
                final_included += 1
                if paper.year:
                    papers_by_year[paper.year] += 1

    years = sorted(papers_by_year.keys())
    year_counts = [papers_by_year[y] for y in years]

    # Exclusion reasons
    exclusion_counts = defaultdict(int)
    for rev in (Review.query.join(Paper)
                .filter(Paper.project_id == pid,
                        Review.decision == "exclude",
                        Review.exclusion_criteria != None,
                        Review.exclusion_criteria != "").all()):
        for cid_str in rev.exclusion_criteria.split(","):
            cid_str = cid_str.strip()
            if cid_str.isdigit():
                exclusion_counts[int(cid_str)] += 1

    criteria_labels = {}
    for cid in exclusion_counts:
        c = Criterion.query.get(cid)
        if c:
            criteria_labels[cid] = f"{c.code}: {c.description[:50]}"

    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    reviewer_stats = []
    for r in reviewers:
        row = {"name": r.name}
        for s in STAGES:
            row[s] = Review.query.filter_by(reviewer_id=r.id, stage=s).join(Paper).filter(Paper.project_id == pid).count()
        reviewer_stats.append(row)

    # --- Reviewer decision breakdown (include/uncertain/exclude %) per stage ---
    reviewer_decisions = []
    for r in reviewers:
        row = {"name": r.name, "stages": {}}
        for s in STAGES:
            revs = (Review.query.filter_by(reviewer_id=r.id, stage=s)
                    .join(Paper).filter(Paper.project_id == pid).all())
            total_s = len(revs)
            if total_s:
                inc = sum(1 for x in revs if x.decision == "include")
                unc = sum(1 for x in revs if x.decision == "uncertain")
                exc = sum(1 for x in revs if x.decision == "exclude")
                row["stages"][s] = {
                    "total": total_s,
                    "include": inc, "include_pct": round(inc / total_s * 100),
                    "uncertain": unc, "uncertain_pct": round(unc / total_s * 100),
                    "exclude": exc, "exclude_pct": round(exc / total_s * 100),
                }
            else:
                row["stages"][s] = None
        reviewer_decisions.append(row)

    # --- Source / database breakdown ---
    source_counts = defaultdict(lambda: {"total": 0, "included": 0})
    last_stage = stages_done[-1] if stages_done else None
    for paper in Paper.query.filter_by(project_id=pid).all():
        src = (paper.source or "Unknown").strip() or "Unknown"
        source_counts[src]["total"] += 1
        if last_stage:
            revs = Review.query.filter_by(paper_id=paper.id, stage=last_stage).all()
            if passes_threshold(revs, project.threshold):
                source_counts[src]["included"] += 1
    source_stats = sorted(source_counts.items(), key=lambda x: -x[1]["total"])

    # --- Uncertain papers still needing a decision ---
    uncertain_papers = {}
    for s in STAGES:
        ups = []
        for paper in get_eligible_papers(pid, s, ignore_pilot=True):
            revs = Review.query.filter_by(paper_id=paper.id, stage=s).all()
            if revs and any(r.decision == "uncertain" for r in revs):
                # Check if there's already a group override
                gd = GroupDecision.query.filter_by(paper_id=paper.id, stage=s).first()
                if not gd or gd.decision == "uncertain":
                    ups.append(paper)
        uncertain_papers[s] = ups

    # --- PRISMA numbers ---
    # Per-stage: imported → title eligible → title included → abstract eligible → …
    prisma = {
        "imported": total,
        "duplicates_removed": 0,   # not tracked yet
        "stages": {}
    }
    for i, s in enumerate(STAGES):
        eligible_papers = get_eligible_papers(pid, s, ignore_pilot=True)
        n_eligible = len(eligible_papers)
        n_included = 0
        n_excluded = 0
        n_uncertain = 0
        for p in eligible_papers:
            revs = Review.query.filter_by(paper_id=p.id, stage=s).all()
            gd = GroupDecision.query.filter_by(paper_id=p.id, stage=s).first()
            effective = gd.decision if gd else (
                compute_consensus([r.decision for r in revs]) if revs else None
            )
            if effective == "include":
                n_included += 1
            elif effective == "exclude":
                n_excluded += 1
            elif effective == "uncertain":
                n_uncertain += 1
        prisma["stages"][s] = {
            "label": STAGE_LABELS[s],
            "eligible": n_eligible,
            "included": n_included,
            "excluded": n_excluded,
            "uncertain": n_uncertain,
        }
    prisma["final_included"] = final_included

    # Source breakdown for PRISMA identification boxes
    prisma["sources"] = [(src, d["total"]) for src, d in source_stats]

    return render_template("statistics.html", project=project,
                           funnel=funnel, years=years, year_counts=year_counts,
                           exclusion_counts=exclusion_counts,
                           criteria_labels=criteria_labels,
                           stages_done=stages_done,
                           final_included=final_included,
                           reviewer_stats=reviewer_stats,
                           reviewer_decisions=reviewer_decisions,
                           source_stats=source_stats,
                           uncertain_papers=uncertain_papers,
                           prisma=prisma)


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/kappa")
def kappa_analysis(pid):
    project = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    stage = request.args.get("stage", "title")
    r1_id = request.args.get("r1", type=int)
    r2_id = request.args.get("r2", type=int)

    kappa_result = None
    agreement_matrix = None
    conflicts = []

    if r1_id and r2_id and r1_id != r2_id:
        r1_map = {r.paper_id: r.decision for r in
                  Review.query.filter_by(reviewer_id=r1_id, stage=stage)
                  .join(Paper).filter(Paper.project_id == pid).all()}
        r2_map = {r.paper_id: r.decision for r in
                  Review.query.filter_by(reviewer_id=r2_id, stage=stage)
                  .join(Paper).filter(Paper.project_id == pid).all()}

        common = set(r1_map) & set(r2_map)
        if len(common) >= 2:
            label_map = {"include": 2, "uncertain": 1, "exclude": 0}
            y1 = [label_map[r1_map[p]] for p in common]
            y2 = [label_map[r2_map[p]] for p in common]
            agreed = sum(1 for p in common if r1_map[p] == r2_map[p])
            simple_pct = round(100 * agreed / len(common), 1)
            degenerate = len(set(y1)) == 1 or len(set(y2)) == 1
            try:
                k    = cohen_kappa_score(y1, y2, labels=[0, 1, 2])
                ac1  = gwets_ac1(y1, y2, n_categories=3)
                kappa_result = {
                    "kappa":              round(k,   4),
                    "ac1":                round(ac1, 4),
                    "n_papers":           len(common),
                    "n_agreed":           agreed,
                    "simple_agreement":   simple_pct,
                    "kappa_interpretation": interpret_agreement(k),
                    "ac1_interpretation":   interpret_agreement(ac1),
                    "degenerate":         degenerate,
                }
                decisions = ["include", "uncertain", "exclude"]
                matrix = [[0] * 3 for _ in range(3)]
                for paper_id in common:
                    i = decisions.index(r1_map[paper_id])
                    j = decisions.index(r2_map[paper_id])
                    matrix[i][j] += 1
                agreement_matrix = matrix

                # Conflicts: papers where decisions differ
                for paper_id in common:
                    if r1_map[paper_id] != r2_map[paper_id]:
                        p = Paper.query.get(paper_id)
                        conflicts.append({
                            "paper": p,
                            "d1": r1_map[paper_id],
                            "d2": r2_map[paper_id],
                        })
            except Exception as e:
                flash(f"Error calculating kappa: {e}", "danger")
        else:
            flash(f"Need at least 2 papers reviewed by both reviewers (found {len(common)}).", "warning")

    r1 = Reviewer.query.get(r1_id) if r1_id else None
    r2 = Reviewer.query.get(r2_id) if r2_id else None
    return render_template("kappa.html", project=project, reviewers=reviewers,
                           stage=stage, r1=r1, r2=r2,
                           kappa_result=kappa_result,
                           agreement_matrix=agreement_matrix,
                           conflicts=conflicts,
                           stages=STAGES, stage_labels=STAGE_LABELS)


# ── Decisions overview (group override) ──────────────────────────────────────

def compute_consensus(decisions):
    """Same logic as the combined export: exclude only if unanimous, include if any included."""
    if not decisions:
        return None
    if all(d == "exclude" for d in decisions):
        return "exclude"
    if any(d == "include" for d in decisions):
        return "include"
    return "uncertain"


@app.route("/project/<int:pid>/decisions/<stage>", methods=["GET", "POST"])
def decisions_overview(pid, stage):
    if stage not in STAGES:
        return redirect(url_for("project_dashboard", pid=pid))
    project   = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()
    papers    = get_eligible_papers(pid, stage, ignore_pilot=True)

    if request.method == "POST":
        for paper in papers:
            val   = request.form.get(f"override_{paper.id}", "")
            notes = request.form.get(f"notes_{paper.id}", "").strip()
            existing = GroupDecision.query.filter_by(
                paper_id=paper.id, stage=stage).first()
            if val in ("include", "uncertain", "exclude"):
                if existing:
                    existing.decision   = val
                    existing.notes      = notes
                    existing.updated_at = datetime.utcnow()
                else:
                    db.session.add(GroupDecision(
                        paper_id=paper.id, stage=stage,
                        decision=val, notes=notes))
            elif val == "" and existing:
                db.session.delete(existing)
        db.session.commit()
        flash("Group decisions saved.", "success")
        return redirect(request.url)

    # Build per-paper display data
    paper_data = []
    for paper in papers:
        rev_map = {r.reviewer_id: r.decision for r in
                   Review.query.filter_by(paper_id=paper.id, stage=stage).all()}
        group      = GroupDecision.query.filter_by(
            paper_id=paper.id, stage=stage).first()
        decisions  = list(rev_map.values())
        consensus  = compute_consensus(decisions)
        has_conflict = len(set(decisions)) > 1 if len(decisions) > 1 else False
        paper_data.append({
            "paper":        paper,
            "rev_map":      rev_map,
            "group":        group,
            "consensus":    consensus,
            "has_conflict": has_conflict,
        })

    return render_template("decisions.html",
                           project=project, stage=stage,
                           stage_label=STAGE_LABELS[stage],
                           reviewers=reviewers,
                           paper_data=paper_data)


# ── Export ────────────────────────────────────────────────────────────────────

@app.route("/project/<int:pid>/export")
def export_papers(pid):
    project = Project.query.get_or_404(pid)
    stage = request.args.get("stage", "fulltext")
    decision_filter = request.args.get("decision", "include")

    rows = []
    for paper in Paper.query.filter_by(project_id=pid).all():
        reviews = Review.query.filter_by(paper_id=paper.id, stage=stage).all()
        if not reviews:
            continue
        counts = defaultdict(int)
        for r in reviews:
            counts[r.decision] += 1
        majority = max(counts, key=counts.__getitem__)
        if decision_filter != "all" and majority != decision_filter:
            continue
        rows.append({
            "Title": paper.title,
            "Authors": paper.authors,
            "Year": paper.year,
            "DOI": paper.doi,
            "Source": paper.source,
            "Abstract": paper.abstract,
            "Decision": majority,
            "Notes": "; ".join(r.notes for r in reviews if r.notes),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(app.config["UPLOAD_FOLDER"], f"export_{pid}_{stage}.csv")
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True,
                     download_name=f"{project.name}_{stage}_{decision_filter}.csv")


# ── Export review decisions ───────────────────────────────────────────────────

DECISION_TO_NUM = {"include": 1, "uncertain": 0.5, "exclude": 0}


@app.route("/project/<int:pid>/export-reviews")
def export_reviews(pid):
    """Per-reviewer export in the importable 0/0.5/1 format."""
    project     = Project.query.get_or_404(pid)
    stage       = request.args.get("stage", "title")
    reviewer_id = request.args.get("reviewer_id", type=int)
    reviewer    = Reviewer.query.get_or_404(reviewer_id)

    if reviewer.project_id != pid:
        return redirect(url_for("manage_reviewers", pid=pid))

    reviews = (Review.query
               .filter_by(reviewer_id=reviewer_id, stage=stage)
               .join(Paper).filter(Paper.project_id == pid)
               .all())

    rows = [{
        "Title":    r.paper.title or "",
        "DOI":      r.paper.doi   or "",
        "Authors":  r.paper.authors or "",
        "Year":     r.paper.year  or "",
        "Decision": DECISION_TO_NUM.get(r.decision, ""),
        "Notes":    r.notes or "",
    } for r in reviews]

    df  = pd.DataFrame(rows)
    out = os.path.join(app.config["UPLOAD_FOLDER"],
                       f"reviews_{pid}_{stage}_{reviewer_id}.csv")
    df.to_csv(out, index=False)
    safe_name = reviewer.name.replace(" ", "_")
    return send_file(out, as_attachment=True,
                     download_name=f"{project.name}_{stage}_{safe_name}_decisions.csv")


@app.route("/project/<int:pid>/export-reviews/all")
def export_reviews_all(pid):
    """Export all stages for one reviewer as a single multi-sheet Excel file."""
    project     = Project.query.get_or_404(pid)
    reviewer_id = request.args.get("reviewer_id", type=int)
    reviewer    = Reviewer.query.get_or_404(reviewer_id)

    if reviewer.project_id != pid:
        return redirect(url_for("assignment_workspace", pid=pid))

    out = os.path.join(app.config["UPLOAD_FOLDER"],
                       f"reviews_{pid}_all_{reviewer_id}.xlsx")
    safe_name = reviewer.name.replace(" ", "_")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        has_data = False
        for stage in ["title", "abstract", "fulltext"]:
            reviews = (Review.query
                       .filter_by(reviewer_id=reviewer_id, stage=stage)
                       .join(Paper).filter(Paper.project_id == pid)
                       .all())
            if not reviews:
                continue
            rows = [{
                "Title":    r.paper.title   or "",
                "DOI":      r.paper.doi     or "",
                "Authors":  r.paper.authors or "",
                "Year":     r.paper.year    or "",
                "Decision": DECISION_TO_NUM.get(r.decision, ""),
                "Notes":    r.notes or "",
            } for r in reviews]
            pd.DataFrame(rows).to_excel(writer, index=False,
                                        sheet_name=stage.capitalize())
            has_data = True

        if not has_data:
            pd.DataFrame(columns=["Title","DOI","Authors","Year","Decision","Notes"])\
              .to_excel(writer, index=False, sheet_name="No decisions")

    return send_file(out, as_attachment=True,
                     download_name=f"{project.name}_{safe_name}_decisions.xlsx")


@app.route("/project/<int:pid>/export-reviews/combined")
def export_reviews_combined(pid):
    """Wide-format export with one column per reviewer + a consensus column."""
    project   = Project.query.get_or_404(pid)
    stage     = request.args.get("stage", "title")
    reviewers = Reviewer.query.filter_by(project_id=pid).all()

    # Collect all papers reviewed in this stage
    reviewed_ids = (db.session.query(Review.paper_id)
                    .join(Paper).filter(Paper.project_id == pid, Review.stage == stage)
                    .distinct().all())
    paper_ids = [r[0] for r in reviewed_ids]

    rows = []
    for paper in Paper.query.filter(Paper.id.in_(paper_ids)).all():
        row = {
            "Title":   paper.title   or "",
            "DOI":     paper.doi     or "",
            "Authors": paper.authors or "",
            "Year":    paper.year    or "",
        }
        numeric_decisions = []
        for rv in reviewers:
            rev = Review.query.filter_by(paper_id=paper.id,
                                         reviewer_id=rv.id, stage=stage).first()
            val = DECISION_TO_NUM[rev.decision] if rev else ""
            row[rv.name] = val
            if rev:
                numeric_decisions.append(DECISION_TO_NUM[rev.decision])

        # Consensus: exclude only if unanimous; include if any included
        if numeric_decisions:
            if all(v == 0 for v in numeric_decisions):
                consensus_num  = 0
                consensus_text = "exclude"
            elif any(v == 1 for v in numeric_decisions):
                consensus_num  = 1
                consensus_text = "include"
            else:
                consensus_num  = 0.5
                consensus_text = "uncertain"
        else:
            consensus_num  = ""
            consensus_text = ""

        row["Consensus_Decision"] = consensus_text
        row["Consensus_Numeric"]  = consensus_num

        group = GroupDecision.query.filter_by(paper_id=paper.id, stage=stage).first()
        row["Group_Override"] = group.decision if group else ""
        row["Group_Notes"]    = group.notes    if group else ""

        rows.append(row)

    df  = pd.DataFrame(rows)
    out = os.path.join(app.config["UPLOAD_FOLDER"],
                       f"combined_{pid}_{stage}.csv")
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True,
                     download_name=f"{project.name}_{stage}_combined_decisions.csv")


# ── Generate assignment export ────────────────────────────────────────────────

@app.route("/project/<int:pid>/assign", methods=["GET", "POST"])
def generate_assignment(pid):
    """Export a blank assignment CSV (or ZIP) for reviewers to fill in offline."""
    project = Project.query.get_or_404(pid)
    reviewers = Reviewer.query.filter_by(project_id=pid).all()

    if request.method == "POST":
        stage              = request.form.get("stage", "title")
        reviewer_ids       = request.form.getlist("reviewer_ids", type=int)
        paper_filter       = request.form.get("paper_filter", "all")   # "all" | "unreviewed"
        reviewers_per_paper = request.form.get("reviewers_per_paper", type=int) or 1

        if not reviewer_ids:
            flash("Select at least one reviewer.", "danger")
            return redirect(request.url)

        n = len(reviewer_ids)
        reviewers_per_paper = max(1, min(reviewers_per_paper, n))

        eligible = get_eligible_papers(pid, stage)

        if paper_filter == "unreviewed":
            reviewed_ids = set()
            for rid in reviewer_ids:
                reviewed_ids |= {r.paper_id for r in
                                 Review.query.filter_by(reviewer_id=rid, stage=stage).all()}
            eligible = [p for p in eligible if p.id not in reviewed_ids]

        if not eligible:
            flash("No papers to assign for the selected options.", "warning")
            return redirect(request.url)

        project_safe = project.name.replace(" ", "_")

        # ── Single reviewer → one xlsx ─────────────────────────────────────────
        if n == 1:
            rv      = Reviewer.query.get(reviewer_ids[0])
            rv_safe = rv.name.replace(" ", "_")
            out     = os.path.join(app.config["UPLOAD_FOLDER"],
                                   f"assign_{pid}_{stage}_{rv_safe}.xlsx")
            _write_assignment_xlsx(out, eligible, pid)
            return send_file(out, as_attachment=True,
                             download_name=f"{project_safe}_{stage}_{rv_safe}_assignment.xlsx")

        # ── Multiple reviewers → ZIP, one xlsx per reviewer ───────────────────
        # Round-robin overlap: paper i is assigned to reviewers at positions
        # [(i + j) % n for j in range(reviewers_per_paper)]
        per_reviewer = {rid: [] for rid in reviewer_ids}
        for i, paper in enumerate(eligible):
            for j in range(reviewers_per_paper):
                per_reviewer[reviewer_ids[(i + j) % n]].append(paper)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rid in reviewer_ids:
                rv      = Reviewer.query.get(rid)
                rv_safe = rv.name.replace(" ", "_")
                tmp = os.path.join(app.config["UPLOAD_FOLDER"],
                                   f"_tmp_assign_{rid}.xlsx")
                _write_assignment_xlsx(tmp, per_reviewer[rid], pid)
                zf.write(tmp, f"{project_safe}_{stage}_{rv_safe}_assignment.xlsx")
                os.remove(tmp)
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         mimetype="application/zip",
                         download_name=f"{project_safe}_{stage}_assignments.zip")

    return render_template("assign.html", project=project, reviewers=reviewers,
                           stages=STAGES, stage_labels=STAGE_LABELS)


# ── Init ──────────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()
    # Migrate: add threshold column to existing projects tables
    from sqlalchemy import text
    try:
        db.session.execute(text(
            "ALTER TABLE projects ADD COLUMN threshold VARCHAR(20) NOT NULL DEFAULT 'any'"
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()   # column already exists — safe to ignore

if __name__ == "__main__":
    app.run(debug=True, port=5001)
