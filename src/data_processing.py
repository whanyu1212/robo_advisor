import pandas as pd
from loguru import logger
from colorama import Fore, init
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

init(autoreset=True)


class DataProcessor:
    def __init__(self, df, useless_columns, categorical_columns, numerical_columns):
        self.df = df.copy()
        self.useless_columns = useless_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def remove_useless_columns(self):
        logger.info(f"{Fore.GREEN}Starting to remove useless columns.")
        df_filtered = self.df.drop(
            columns=[col for col in self.useless_columns if col in self.df.columns]
        )
        logger.info(f"{Fore.GREEN}Finished removing useless columns.")
        return df_filtered

    def encode_categorical_columns(self, df_filtered):
        dummy_frames = []
        for col in self.categorical_columns:
            dummy_frame = pd.get_dummies(df_filtered[col], prefix=col)
            dummy_frames.append(dummy_frame)
            df_dummies = pd.concat(dummy_frames, axis=1)

        return df_dummies

    def combine_dummy_n_numeric(
        self, df_dummies: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> pd.DataFrame:
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
        # df_final.to_csv("./data/df_cleaned.csv", index=False)
        return df_final
