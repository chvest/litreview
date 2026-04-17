# LitReview

A local web application for conducting systematic literature reviews following [Kitchenham's guidelines](https://www.cs.ecu.edu/gudap/Teaching/CS6360/Systematic%20Review%20Guidelines.pdf). Runs entirely on your own machine — no cloud account or internet connection required during use.

---

## Features

- **Import papers** from CSV or Excel (PubMed, Scopus, Web of Science, etc.) with a column-mapping step
- **Three-stage screening** — title, abstract, and full-text review
- **Randomised paper order** per reviewer to minimise ordering bias
- **Inclusion & exclusion criteria** panel visible during every review stage
- **Pilot review workflow** — calibrate reviewers on a shared sample before full screening
- **Abstract term highlighting** — keywords from your criteria are highlighted in the abstract text
- **Multiple reviewers** — each reviewer gets an independent randomised order
- **Cohen's κ** inter-rater agreement analysis with conflict list
- **Import existing reviews** from CSV/Excel (0 = exclude, 0.5 = uncertain, 1 = include)
- **Export decisions** — per-reviewer in importable format, or combined wide-format with consensus
- **Statistics page** — screening funnel, papers by publication year, exclusion reason breakdown
- **Dark mode** toggle

---

## Requirements

- **Python 3.10 or newer** — download from [python.org](https://www.python.org/downloads/)
  - On Windows: tick **"Add Python to PATH"** during installation

That's it. No database server, no Node.js, no other dependencies.

---

## Setup (first time only)

```bash
# 1. Clone or download the repository
git clone https://github.com/chvest/litreview.git
cd litreview

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Then open **http://localhost:5001** in your browser.

The database (`litreview.db`) is created automatically on first run.

> **Windows note:** if `python` is not found, try `py app.py` and `py -m pip install -r requirements.txt` instead.

---

## Running after first setup

```bash
python app.py
```

Then open **http://localhost:5001**.

---

## Multi-reviewer workflow

The app is designed to run on a single machine. For distributed reviewing:

1. Each reviewer runs their own local copy of the app and works independently
2. When done, they export their decisions via **Reviewers → Export → [stage]**
3. The lead reviewer imports each file via **Import → Import Reviews**
4. Use the **Cohen's κ** page to measure inter-rater agreement
5. The **Export → Combined reviewer decisions** download gives a single wide-format CSV with one column per reviewer and a consensus column

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Use `py app.py` (Windows) or `python3 app.py` (macOS/Linux) |
| `pip` not found | Use `py -m pip install -r requirements.txt` |
| Port 5001 already in use | Change `port=5001` at the bottom of `app.py` |
| `ModuleNotFoundError` | Re-run the pip install step |
