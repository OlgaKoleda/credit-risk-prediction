# Credit Risk Prediction

Predicting which borrowers will get into serious financial trouble within two years, and turning the risk score into a lending decision, using the [Kaggle "Give Me Some Credit" competition](https://www.kaggle.com/c/GiveMeSomeCredit) (150,000 borrowers).

> 🚧 Work in progress: exploration and a first model are done.

![Model comparison](images/model_comparison.png)

## Notebooks

| Notebook | What it covers |
|---|---|
| [01_exploration](notebooks/01_exploration.ipynb) | A rare target (6.7%), sanity checks for copies, special codes, outliers, a column with two meanings, what predicts trouble |
| [02_first_model](notebooks/02_first_model.ipynb) | Baselines, logistic regression vs. gradient boosting, the effect of cleaning, choosing a decision threshold from the costs of mistakes, first Kaggle submission |

## Key findings so far

- **Past behaviour predicts future behaviour.** Borrowers never late in the last two years default at 2.8%; those late at least once at 22%.
- **The data has traps.** Late-payment counts of 96 and 98 are codes, not counts. Credit utilisation reaches 50,708 times the limit. `DebtRatio` is a ratio when the income is known, but a dollar amount when it is missing.
- **Cleaning was worth 0.15 AUC for logistic regression** (0.70 → 0.855) and nothing for gradient boosting. Linear models need clean data; tree models mostly do not.

| Model (5-fold cross-validation) | ROC AUC |
|---|---|
| Rule: "late at least once" | 0.746 |
| Logistic regression, raw data | 0.698 |
| Logistic regression, cleaned + log | 0.855 |
| **Gradient boosting, cleaned** | **0.864** |

- **The threshold is a business decision.** With illustrative costs (a missed defaulter costs 5 times as much as a rejected good borrower), the cheapest rule is to reject applicants above 17% risk, not 50%. It catches 56% of defaulters instead of 18%, and cuts the cost of mistakes by about a quarter.

![Cost by threshold](images/threshold_cost.png)

| Kaggle submission | Model | CV ROC AUC | Public | Private |
|---|---|---|---|---|
| v1 | Gradient boosting, cleaned features | 0.8644 | 0.86024 | **0.86752** |

## Project structure

```
├── data/                  # Kaggle files (not committed – see below)
├── images/                # Charts saved by the notebooks
├── notebooks/             # Numbered notebooks, run in order
├── src/
│   └── features.py        # CreditFeatures: the cleaning decisions from notebook 01
├── submissions/           # Kaggle prediction files
├── requirements.txt
└── README.md
```

## How to run

1. Download the competition files from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) into the `data/` folder (`cs-training.csv`, `cs-test.csv`).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in `notebooks/` in VS Code or Jupyter and run them in order.

## Tools

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · Jupyter
