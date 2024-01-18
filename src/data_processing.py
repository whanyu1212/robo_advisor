from typing import List

import pandas as pd
from colorama import Fore, init
from loguru import logger
from sklearn.preprocessing import LabelEncoder

init(autoreset=True)


class DataProcessor:
    """
    A class used to process data for machine learning.

    ...

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to be processed.
    useless_columns : list
        A list of column names to be removed from the DataFrame.
    categorical_columns : list
        A list of names of categorical columns to be encoded.
    numerical_columns : list
        A list of names of numerical columns to be included in the final DataFrame.

    Methods
    -------
    remove_useless_columns():
        Removes specified columns from the DataFrame.
    encode_categorical_columns(df_filtered):
        Encodes categorical columns using one-hot encoding.
    combine_dummy_n_numeric(df_dummies, df_filtered):
        Combines dummy (encoded) and numerical columns into a final DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        useless_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ):
        self.df = df.copy()
        self.useless_columns = useless_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def remove_useless_columns(self) -> pd.DataFrame:
        """
        Based on the list of useless columns, removes them from the
        DataFrame.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        logger.info(f"{Fore.GREEN}Starting to remove useless columns.")
        df_filtered = self.df.drop(
            columns=[col for col in self.useless_columns if col in self.df.columns]
        )
        logger.info(f"{Fore.GREEN}Finished removing useless columns.")
        return df_filtered

    def encode_categorical_columns(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encode all the categorical columns. Join the encoded
        columns with the numerical columns and response the original
        DataFrame.

        Args:
            df_filtered (pd.DataFrame): df after removal of useless columns

        Raises:
            KeyError: if the categorical column does not exist in the DataFrame

        Returns:
            pd.DataFrame: encoded DataFrame
        """
        dummy_frames = []
        for col in self.categorical_columns:
            if col not in df_filtered.columns:
                raise KeyError(f"Column {col} does not exist in the DataFrame")
            dummy_frame = pd.get_dummies(df_filtered[col], prefix=col)
            dummy_frames.append(dummy_frame)
        df_dummies = pd.concat(dummy_frames, axis=1)
        return df_dummies

    def combine_dummy_n_numeric(
        self, df_dummies: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine dummy and numerical columns into a final DataFrame.
        Label encode the response variable.

        Args:
            df_dummies (pd.DataFrame): dataframe with encoded
            categorical columns
            df_filtered (pd.DataFrame): original dataframe after
            removal of useless columns

        Returns:
            pd.DataFrame: processed dataframe
        """
        df_final = pd.concat(
            [
                df_dummies,
                df_filtered[self.numerical_columns],
                df_filtered["Investment_Strategy"],
            ],
            axis=1,
        )
        le = LabelEncoder()
        le.fit(df_final.Investment_Strategy)
        df_final.Investment_Strategy = le.transform(df_final.Investment_Strategy)
        logger.info(f"{Fore.GREEN}Finished combining dummy and numeric columns.")
        return df_final
