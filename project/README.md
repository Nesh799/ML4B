# Activity Classifier – Sitzen / Stehen / Gehen / Chillen

Local Python project to train a classifier on your own collected smartphone sensor CSV data and predict activities:

- Sitzen
- Stehen
- Gehen
- Chillen

## Setup

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
cd project
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Data format

Put your CSV at `data/raw_data.csv` (or pass a different path via `--data`).

Requirements:

- Your label column must be named one of: `label`, `activity`, `class`, `target` (case-insensitive)
- All numeric columns are treated as sensor features
- Missing values are handled automatically

## Train a model

```bash
python -m src.train --data data/raw_data.csv --out models/model.pkl
```

This will:

- auto-detect the label column
- detect numeric sensor columns
- impute missing values
- scale features (StandardScaler)
- train a RandomForestClassifier
- save to `models/model.pkl`

## Predict on new data

```bash
python -m src.predict --model models/model.pkl --csv data/new_data.csv
```

Optionally save predictions:

```bash
python -m src.predict --model models/model.pkl --csv data/new_data.csv --out data/predictions.csv
```

## Streamlit web app

```bash
streamlit run web/app.py
```

The app will:

- load `models/model.pkl` automatically
- preprocess uploaded CSV using the same training preprocessing
- show predictions + charts (distribution + timeline)

## Data Transformer

To test the model itself, you need to use the app “Sensor Logger” to record the gyroscope and the accelerometer. From the two CSV files, you can then generate a single CSV file that the model can read. Use the AccAndGyroTransformer.py file for this it generates the data needed for the model itself.

```
python AccAndGyroTransformer.py [Accelerometer.csv] [Gyroscope.csv] [Ausgabedatei.csv] --activity ["sitzen"/"stehen"/"gehen"/"chillen"]
```

