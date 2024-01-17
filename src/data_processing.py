import numpy as np
import pandas as pd
from scipy import stats


class DataProcessor:
    def __init__(self, df, useless_columns, categorical_columns, numerical_columns):
        self.df = df.copy()
        self.useless_columns = useless_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def remove_useless_columns(self):
        for column in self.useless_columns:
            if column in self.df.columns:
                self.df = self.df.drop(columns=column)
        return self.df

    def encode_categorical_columns(self):
        for column in self.categorical_columns:
            if column in self.df.columns and column != "Investment_Strategy":
                dummies = pd.get_dummies(self.df[column], prefix=column)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(column, axis=1)
        return self.df

    def outlier_removal(self):
        for column in self.numerical_columns:
            if column in self.df.columns:
                z = np.abs(stats.zscore(self.df[column]))
                self.df = self.df[z < 3]
        return self.df
