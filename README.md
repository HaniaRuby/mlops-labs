# mlopslabs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps Labs — Titanic Survival Prediction with End-to-End Tracking and Online Serving.

---

## Project Organization


```

├── LICENSE             <- Open-source license if one is chosen
├── Makefile            <- Makefile with convenience commands like `make data` or `make train`
├── README.md           <- The top-level README for developers using this project.
├── Dockerfile          <- Multi-stage optimized Docker deployment container utilizing uv cache mounts
├── server.py           <- Application entrypoint script hosting the LitServe engine
├── predict.py          <- Validation script verifying dynamic remote MLflow registry version tracking
├── dvc.yaml            <- Pipeline stage definitions handling data tracking and reproduction gates
├── pyproject.toml      <- Project configuration file with package metadata and dependencies managed by uv
│
├── data
│   ├── external        <- Data from third party sources.
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump.
│
├── docs                <- A default mkdocs project; see www.mkdocs.org for details
│
├── models              <- Trained and serialized local fallback model artifacts managed by DVC
│
├── notebooks           <- Jupyter notebooks for experimentation and analysis.
│
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
│
└── mlopslabs           <- Source code for use in this project.
│
├── **init**.py     <- Makes mlopslabs a Python module
│
├── config.py       <- Store useful variables and configuration
│
├── dataset.py      <- Pipeline runner handling MLflow instrumentation and remote metric logging
│
├── features.py     <- Code to create features for modeling
│
├── modeling

│   ├── **init**.py
│   ├── predict.py  <- Code to run model inference with trained models

│   └── train.py    <- Code to construct and train pipelines
│
└── deployment      <- Online inference architecture implementations
└── online
├── **init**.py
├── api.py       <- LitServe core engine tracking Pydantic validations and batch parsing
└── requests.py  <- Pydantic contract definition layer for batch-capable schema handling

```

---

## MLflow Tracking & DagsHub Model Registry

To preserve pipeline isolation and bypass local file-locking conflicts introduced by DVC, model registration streams the serialized scikit-learn training object directly to DagsHub's hosted tracking URI over memory buffers.

### Reproducing the Execution Pipeline
To trigger a clean re-run of your tracking environment, ensure your workspace environment variables are explicitly populated before prompting DVC execution:

```bash
# Set your active authentication vectors
export DAGSHUB_TOKEN="your_actual_token_here"
export MLFLOW_TRACKING_USERNAME="haniaruby"
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

# Force pipeline stage execution
uv run dvc repro -f train

```

### Verifying Remote Registry Identity

The tracking configuration registers artifacts directly to the dashboard asset ledger under `TitanicSurvivalModel`. You can run the remote testing script to pull versioned registry files down dynamically from the tracking storage layer:

```bash
uv run python predict.py

```

---

## Online Batch Serving with LitServe

The production deployment architecture utilizes **LitServe** (built by Lightning AI on top of FastAPI) to spin up an asynchronous high-performance server. The decoding pipeline maps inputs into Pydantic validated data structures capable of processing multiple user rows inside a single payload request array.

### Starting the Web Server

Launch the server engine from your root workspace terminal:

```bash
uv run python server.py

```

The application will dynamically resolve local binary states and initiate listeners on default cluster port `8000`.

![API Response in Bruno](./api response screenshot.png)

### Batch Testing Configuration via Curl / Bruno

To test multi-record parsing functionality, issue a `POST` request to the `/predict` endpoint using the following sample array layout:

**Endpoint:** `POST http://127.0.0.1:8000/predict`

**Payload Schema:**

```json
{
  "input": [
    {
      "Pclass": 3,
      "Age": 22.0,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 7.25,
      "Sex": "male",
      "Embarked": "S"
    },
    {
      "Pclass": 1,
      "Age": 38.0,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 71.2833,
      "Sex": "female",
      "Embarked": "C"
    }
  ]
}

```

### Containerized Deployment

The provided `Dockerfile` leverages multi-stage builds and bind mounts over `ghcr.io/astral-sh/uv` to build production containers safely:

```bash
# Build the container image
docker build -t mlopslabs-serving:latest .

# Run the localized application layer container
docker run -p 8000:8000 mlopslabs-serving:latest

```