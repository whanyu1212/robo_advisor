import random
import string

import numpy as np
import pandas as pd
from pandas import Series

from src.utils.strategic_conditions import strategy_conditions


def assign_strategy(row: Series, strategy_conditions: list) -> str:
    """
    Assign value for Investment_Strategy column based on the conditions
    defined in strategy_conditions.

    Args:
        row (Series): each row of the dataframe
        strategy_conditions (list): list of conditions and corresponding
        strategies pre-defined

    Returns:
        str: value for Investment_Strategy column assigned based on the conditions
    """
    for strategy_condition in strategy_conditions:
        if all(condition(row) for condition in strategy_condition["conditions"]):
            return strategy_condition["strategy"]

    return (
        "Balanced"
        if np.random.rand() < 0.9
        else np.random.choice(
            [s["strategy"] for s in strategy_conditions if s["strategy"] != "Balanced"]
        )
    )


def generate_user_id_str(n_samples: int, seed: int = 0) -> list[str]:
    """
    Generate a list of random user ids.

    Args:
        n_samples (int): The number of user ids to generate.
        seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
        list[str]: A list of random user ids.
    """
    random.seed(seed)
    return [
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        for _ in range(n_samples)
    ]


def generate_age(n_samples: int) -> list[int]:
    """
    Generate a list of random ages between 18 and 70.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random ages
    """
    return np.random.randint(18, 70, n_samples)


def generate_income_level(n_samples: int) -> list[str]:
    """
    Generate a list of random income levels.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random income levels
    """
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_credit_score(n_samples: int) -> list[int]:
    """
    Generate a list of random credit scores.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random credit scores
    """
    return np.random.normal(700, 50, n_samples).astype(int)


def generate_investment_experience(n_samples: int) -> list[str]:
    """
    Generate a list of random investment experience levels.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random investment experience levels
    """
    return np.random.choice(["Novice", "Intermediate", "Experienced"], n_samples)


def generate_financial_knowledge(n_samples: int) -> list[str]:
    """
    Generate a list of random financial knowledge levels.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random financial knowledge levels
    """
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_risk_tolerance(n_samples: int) -> list[str]:
    """
    Generate a list of random risk tolerance levels.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random risk tolerance levels
    """
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_investment_goals(n_samples: int) -> list[str]:
    """
    Generate a list of random investment goals.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random investment goals
    """
    return np.random.choice(
        ["Preservation", "Income", "Growth", "Speculation"], n_samples
    )


def generate_time_horizon(n_samples: int) -> list[str]:
    """
    Generate a list of random time horizons.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[str]: List of random time horizons
    """
    return np.random.choice(["Short-term", "Medium-term", "Long-term"], n_samples)


def generate_total_assets(n_samples: int) -> list[float]:
    """
    Generate a list of random total assets.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random total assets
    """
    return np.random.normal(5000000, 150000, n_samples)


def generate_total_liabilities(n_samples: int) -> list[float]:
    """
    Generate a list of random total liabilities.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random total liabilities
    """
    return np.random.normal(2000000, 100000, n_samples)


def generate_number_of_dependents(n_samples: int) -> list[int]:
    """
    Generate a list of random number of dependents.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random number of dependents
    """
    return np.random.randint(0, 5, n_samples)


def generate_years_of_investing(n_samples: int) -> list[int]:
    """
    Generate a list of random years of investing.

    Args:
        n_samples (int): number of samples specified

    Returns:
        list[int]: List of random years of investing
    """
    return np.random.randint(0, 40, n_samples)


def generate_synthetic_data(n_samples: int = 30000, seed: int = 0) -> pd.DataFrame:
    """
    Generate a synthetic dataset by calling the above functions.

    Args:
        n_samples (int, optional): Sample size specified. Defaults to 30000.
        seed (int, optional): Seed for random number generation. Defaults to 0.

    Raises:
        ValueError: Check if n_samples is a positive integer

    Returns:
        pd.DataFrame: A synthetic dataset for ml model training
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    try:
        np.random.seed(seed)
        data = pd.DataFrame(
            {
                "User_ID": generate_user_id_str(n_samples),
                "Age": generate_age(n_samples),
                "Income_Level": generate_income_level(n_samples),
                "Credit_Score": generate_credit_score(n_samples),
                "Investment_Experience": generate_investment_experience(n_samples),
                "Financial_Knowledge": generate_financial_knowledge(n_samples),
                "Risk_Tolerance": generate_risk_tolerance(n_samples),
                "Investment_Goals": generate_investment_goals(n_samples),
                "Time_Horizon": generate_time_horizon(n_samples),
                "Total_Assets": generate_total_assets(n_samples),
                "Total_Liabilities": generate_total_liabilities(n_samples),
                "Number_of_Dependents": generate_number_of_dependents(n_samples),
                "Years_of_Investing": generate_years_of_investing(n_samples),
            }
        )

        # Calculating Net Worth
        data["Net_Worth"] = data["Total_Assets"] - data["Total_Liabilities"]

        # Assigning Investment Strategies
        data["Investment_Strategy"] = data.apply(
            lambda row: assign_strategy(row, strategy_conditions), axis=1
        )

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# test the above functions
if __name__ == "__main__":
    data = generate_synthetic_data()
    print(data.head())
    data.to_csv("./data/raw/synthetic_data.csv", index=False)
