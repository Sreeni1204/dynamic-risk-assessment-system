# dynamic-risk-assessment-system

## Overview

This project implements a dynamic risk assessment system for business process automation. It includes data ingestion, model training, scoring, deployment, diagnostics, and reporting, all orchestrated via a Flask API and automation scripts.

---

## Project Structure

- `dynamic_risk_assessment_system/` — Main package (Flask app, model, diagnostics, etc.)
- `data/` — Source, practice, and test data
- `models/` — Model files, deployment, and scores
- `process_automation/` — Automation scripts (`fullprocess.py`, `apicalls.py`)
- `config.json` — Configuration file for paths
- `pyproject.toml` — Poetry project configuration

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sreeni1204/dynamic-risk-assessment-system.git
   cd dynamic-risk-assessment-system
   ```

2. **Install dependencies with Poetry v2.1.4**
   ```bash
   poetry install
   ```

3. **(Optional) Activate virtual environment**
   ```bash
   poetry shell
   ```

---

## Running the Flask App

```bash
dynamic_risk_assessment_system --config_path=config.json --host=localhost --port=8000
```

The app will be available at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

- `GET /first_run_on_practice_data` — Run full pipeline on practice data
- `POST /prediction` — Make predictions (send test data as JSON)
- `GET /scoring` — Get model F1 score
- `GET /summarystats` — Get summary statistics for numerical columns
- `GET /diagnostics` — Get diagnostics (missing data, execution time, outdated packages)

### Example: Calling the Prediction Endpoint

```python
import requests
import pandas as pd

test_df = pd.read_csv("data/testdata/testdata.csv")
test_json = test_df.to_dict(orient='records')
response = requests.post("http://localhost:8000/prediction", json=test_json)
print(response.json())
```

---

## Automation

### Full Process Automation

Run the full automation pipeline:
```bash
poetry run python process_automation/fullprocess.py
```

### API Calls Automation

Call all endpoints and save responses:
```bash
poetry run python process_automation/apicalls.py
```

### Cronjob Example

To schedule the cronjob for process automation, first step is to install the poetry package

Second step is to run the flask app installed via poetry package using below command

```sh
dynamic_risk_assessment_system --config_path=config.json --host=localhost --port=8000
```

Once the app is up and running, run the cronjob.

To automate `fullprocess.py` via cron, add to your crontab:
```
10 0 * * * cd /home/sreeni/Udacity/MLOps/Projects/dynamic-risk-assessment-system && poetry run python process_automation/fullprocess.py
```

---

## Configuration

Edit `config.json` to set paths for input data, output models, test data, and deployment.

---

## Contact

For questions, contact [hvsreenivasa93@gmail.com](mailto:hvsreenivasa93@gmail.com)