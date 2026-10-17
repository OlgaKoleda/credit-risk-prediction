"""Cleaning and feature engineering for the Give Me Some Credit data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "SeriousDlqin2yrs"
LATE_COLUMNS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"]
SPECIAL_CODES = [96, 98]   # not real counts: these customers have no credit history (see notebook 01)


def load(path):
    """Read a Kaggle file and name the unnamed first column Id."""
    return pd.read_csv(path).rename(columns={"Unnamed: 0": "Id"})


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Fix the data problems found in notebook 01 and add a few features.

    - The late-payment counts 96 and 98 are codes, not counts: they are replaced by a flag.
    - DebtRatio is a ratio only when the income is known; otherwise it is a dollar amount.
      The two meanings are split into two columns.
    - Very large values (utilisation over 10, impossible ages) are capped.
    - Missing income gets its own yes/no column, so the model can learn from the gap itself.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=["Id", TARGET], errors="ignore").copy()

        special = X[LATE_COLUMNS].isin(SPECIAL_CODES).any(axis=1)
        X["NoCreditHistoryCode"] = special.astype(int)
        X[LATE_COLUMNS] = X[LATE_COLUMNS].mask(X[LATE_COLUMNS].isin(SPECIAL_CODES), 0)
        X["TimesLate"] = X[LATE_COLUMNS].sum(axis=1)

        income_missing = X["MonthlyIncome"].isna()
        X["IncomeMissing"] = income_missing.astype(int)
        X["MonthlyDebtWhenIncomeUnknown"] = X["DebtRatio"].where(income_missing, 0)
        X["DebtRatio"] = X["DebtRatio"].where(~income_missing)          # NaN when it is not a ratio

        X["RevolvingUtilizationOfUnsecuredLines"] = X["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10)
        X["age"] = X["age"].clip(lower=18)
        return X


def raw_columns(X):
    """The columns exactly as in the file, without Id and target (to compare with CreditFeatures)."""
    return X.drop(columns=["Id", TARGET], errors="ignore")


def log_positive(X):
    """log(1 + x) for every value, with negative values treated as 0."""
    return np.log1p(np.clip(X, 0, None))
