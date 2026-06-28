# FakeNewsDetection — Agent Guide

## Entrypoints

| File | What it is | How to run |
|---|---|---|
| `FakeNews.ipynb` | Jupyter notebook (PassiveAggressiveClassifier, ipywidgets UI) | `jupyter notebook FakeNews.ipynb` |
| `FakeNews.py` | Python script (5 models compared, ipywidgets UI) | `python FakeNews.py` |
| `mini.py` | Streamlit web app (PassiveAggressiveClassifier) | `streamlit run mini.py` |

## Data

- `news.csv` must be in the repo root at runtime.
- `mini.py:106` has a **hardcoded absolute path** (`/home/pavani/majorproject/news.csv`) — change it to `"news.csv"` before running locally.
- Columns: `title`, `text`, `label`.

## Dependencies

No `requirements.txt` or `pyproject.toml`. Install manually:

```
pip install pandas numpy scikit-learn streamlit ipywidgets jupyter
```

## Repo conventions

- Flat layout — no packages, no `__init__.py`, no tests.
- No linting, typechecking, or CI.
- Labels in `news.csv` are `REAL`/`FAKE` — confirmed from notebook prediction logic (`FakeNews.ipynb:109`).
