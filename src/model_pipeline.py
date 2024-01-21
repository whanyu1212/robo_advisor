import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


class ModelPipeline:
    """A pipeline for training and evaluating a LightGBM model."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initializes the ModelPipeline with a DataFrame, target column,
        test size, and random state.

        Args:
            df (pd.DataFrame): The DataFrame to use for training and testing.
            target_column (str): The name of the target column in the DataFrame.
            test_size (float, optional): Defaults to 0.2.
            random_state (int, optional): Defaults to 42.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self) -> None:
        """
        Splits the DataFrame into training and testing sets.

        The function uses the sklearn's train_test_split function
        to split the DataFrame into training and testing sets.
        The split is based on the test_size and random_state
        parameters provided during the class initialization.
        The target column is dropped from the training set and
        used as the target for the testing set.

        Returns:
            None
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def fit_model(self) -> None:
        """
        Fits a LightGBM model to the training data.

        The function initializes a LightGBM classifier with the
        random state provided during class initialization,
        then fits the model to the training data.

        Returns:
            None
        """
        self.model = LGBMClassifier(random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

    def eval_model(self) -> None:
        """
        Evaluates the model on the testing data.

        The function calculates the score of the model on
        the testing data and prints the result.

        Returns:
            None
        """
        print(self.model.score(self.X_test, self.y_test))

    def calculate_feature_importance(self, n: int) -> pd.Series:
        """
        Calculates the feature importances of the model.

        The function calculates the feature importances of the model,
        sorts them in descending order,
        and returns the top n features.

        Args:
            n (int): The number of top features to return.

        Returns:
            pd.Series: A series of the top n feature importances,
            sorted in descending order.
        """
        feature_importance = pd.Series(
            self.model.feature_importances_, index=self.X_train.columns
        )
        return feature_importance.sort_values(ascending=False).head(n)

    def run(self, n: int) -> pd.Series:
        """
        Runs the entire pipeline.

        The function splits the data, fits the model, evaluates the model,
        and calculates the feature importances, then returns the top n feature
        importances.

        Args:
            n (int): The number of top features to return.

        Returns:
            pd.Series: A series of the top n feature importances, sorted
            in descending order.
        """
        self.split_data()
        self.fit_model()
        self.eval_model()
        return self.calculate_feature_importance(n)
