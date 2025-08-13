# NBA Analytics: Playoff Modeling and Matchup Simulator

[Open in Colab](https://colab.research.google.com/github/PCDAONE/nba-analytics/blob/main/Tuned_NBA_Playoff_Model.ipynb)

## Overview

This project builds machine learning models to estimate NBA playoff performance from team advanced statistics and provides simple matchup simulations. Using seven seasons of team-level data, we train gradient-boosted tree regressors to predict:

- Playoff win percentage
- Total playoff wins

The notebook explores feature correlations, tunes hyperparameters, evaluates models, and includes interactive cells to compare two teams by year.

## Data

- **Source**: Team advanced statistics from [Basketball-Reference](https://www.basketball-reference.com/)
- **Files**: Season-level CSVs included in this repo:
  - `2014-2015 - Sheet1.csv`
  - `2015-2016 - Sheet1.csv`
  - `2016-2017 - Sheet1.csv`
  - `2017-2018 - Sheet1.csv`
  - `2018-2019 - Sheet1.csv`
  - `2019-2020 - Sheet1.csv`
  - `2020-2021 - Sheet1.csv`
  - `2021-2022 - Sheet1.csv` (held out for current-season predictions in the notebook)
- **Labels** (manually added in the notebook):
  - `Playoff Wins`
  - `Playoff Win Percentage`

### Preprocessing (performed in the notebook)

- Add a `Year` column to each season file and concatenate seasons
- Drop unnamed helper columns and non-numeric fields used only for display
- Filter to playoff teams (rows where the `Team` field contains an asterisk `*`)
- Build numeric-only dataset for modeling (`non_str_data.csv`)
- Persist intermediate artifacts (`metadata.csv`, `non_str_data.csv`)

Example columns used (subset): `Age, W, L, PW, PL, MOV, SOS, SRS, ORtg, DRtg, NRtg, Pace, FTr, 3PAr, TS%, eFG%, TOV%, ORB%, FT/FGA, deFG%, dTOV%, dDRB%, dFT/FGA`.

## Modeling

- **Algorithm**: XGBoost Regressor (`xgboost.XGBRegressor`)
- **Targets**: (1) playoff win percentage, (2) playoff wins
- **Train/test split**: 75%/25% with `random_state=40`
- **Tuning**: `RandomizedSearchCV` over depth, learning rate, subsample, colsample parameters (5-fold CV, 25 trials)
- **Chosen example configuration** (from notebook):
  - `subsample=0.6, n_estimators=500, max_depth=3, learning_rate=0.01, colsample_bytree=0.7, colsample_bylevel=0.6`

### Evaluation (illustrative, from a sample run in the notebook)

- Playoff win percentage model: MSE ≈ `0.0184` on the test split
- Playoff wins model: MSE ≈ `7.96` on the test split

The notebook also:

- Plots a correlation heatmap and scatter matrix of features
- Visualizes XGBoost feature importances for both targets

## Matchup Simulation (Notebook)

The notebook includes simple interactive simulations:

1) Historical matchup (any of the combined seasons)
- Enter keys as `Team*Year`, for example:
  - `Golden State Warriors*2016`
  - `Cleveland Cavaliers*2016`
- The model compares predicted playoff win percentage and prints a winner.

2) Current-season matchup (using `2021-2022 - Sheet1.csv`)
- Enter keys like `Golden State Warriors*2022` and `Brooklyn Nets*2022`
- The notebook predicts the winner and, with the second model, predicted total playoff wins for the team.

Note: The notebook also contains a toy “random-stat” game that samples a single stat and awards a point to the higher value. This is for illustration only and is not an accurate predictor.

## How to Run

### Option A: Colab (recommended)

1. Click the link above to open `Tuned_NBA_Playoff_Model.ipynb` in Colab.
2. Upload the CSVs into the Colab runtime or mount Google Drive.
3. If needed, adjust the file paths in the notebook (paths like `/content/2019-2020 - Sheet1.csv`) to match where you placed the files.
4. Runtime → Run all.

### Option B: Local

1. Create and activate a Python environment (Python 3.8+ recommended).
2. Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn statsmodels scipy requests
```

3. Launch Jupyter and open the notebook:

```bash
jupyter lab  # or: jupyter notebook
```

4. Update the file paths in the data-loading cells to point to the CSVs in this repository (instead of `/content/...`).
5. Run all cells.

## Repository Contents

- `Tuned_NBA_Playoff_Model.ipynb` — end-to-end data prep, modeling, evaluation, and simulations
- `2014-2015 - Sheet1.csv` … `2021-2022 - Sheet1.csv` — season datasets from Basketball-Reference (team advanced stats)
- `LICENSE` — MIT License

Artifacts written by the notebook at runtime:

- `metadata.csv` — combined dataset across seasons with manual labels
- `non_str_data.csv` — numeric-only dataset used to train the models

## Notes and Assumptions

- Manual labels for playoff outcomes were added inside the notebook.
- Random seeds are set where possible, but some algorithms (e.g., XGBoost) may still exhibit small run-to-run variance.
- The models are trained on past seasons and do not incorporate roster changes, injuries, schedules, or playoff brackets.

## Acknowledgements

- Data: [Basketball-Reference](https://www.basketball-reference.com/)

## License

This project is released under the MIT License. See `LICENSE` for details.
