from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Tuple

class GBModel:
    def __init__(self, params: dict = None):
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }
        self.model = GradientBoostingRegressor(**self.params)

    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for gradient boosting model
        """
        X = df[feature_cols]
        y = df[target_col]
        return train_test_split(X, y, test_size=0.2, shuffle=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the gradient boosting model
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        """
        return self.model.predict(X) 