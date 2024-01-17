import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


class ModelPipeline:
    def __init__(self, df, target_column, test_size=0.2, random_state=42):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def fit_model(self):
        self.model = LGBMClassifier(random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

    def calculate_feature_importance(self, n):
        feature_importance = pd.Series(
            self.model.feature_importances_, index=self.X_train.columns
        )
        return feature_importance.sort_values(ascending=False).head(n)

    def run(self, n):
        self.split_data()
        self.fit_model()
        return self.calculate_feature_importance(n)
