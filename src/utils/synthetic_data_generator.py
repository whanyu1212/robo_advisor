import random

import numpy as np
import pandas as pd

from src.utils.strategic_conditions import strategy_conditions


def assign_strategy_based_on_conditions(row, conditions, strategy, strategies):
    if all(condition(row) for condition in conditions):
        return (
            strategy
            if random.random() < 0.9
            else random.choice([s for s in strategies if s != strategy])
        )
    return None


def assign_strategy(row, strategy_conditions):
    for strategy_condition in strategy_conditions:
        if all(
            condition(row) for condition in strategy_condition["conditions"]
        ):
            return strategy_condition["strategy"]

    return (
        "Balanced"
        if np.random.rand() < 0.9
        else np.random.choice(
            [
                s["strategy"]
                for s in strategy_conditions
                if s["strategy"] != "Balanced"
            ]
        )
    )


def generate_age(n_samples):
    return np.random.randint(18, 70, n_samples)


def generate_income_level(n_samples):
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_credit_score(n_samples):
    return np.random.normal(700, 50, n_samples).astype(int)


def generate_investment_experience(n_samples):
    return np.random.choice(
        ["Novice", "Intermediate", "Experienced"], n_samples
    )


def generate_financial_knowledge(n_samples):
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_risk_tolerance(n_samples):
    return np.random.choice(["Low", "Medium", "High"], n_samples)


def generate_investment_goals(n_samples):
    return np.random.choice(
        ["Preservation", "Income", "Growth", "Speculation"], n_samples
    )


def generate_time_horizon(n_samples):
    return np.random.choice(
        ["Short-term", "Medium-term", "Long-term"], n_samples
    )


def generate_total_assets(n_samples):
    return np.random.normal(500000, 150000, n_samples).astype(int)


def generate_total_liabilities(n_samples):
    return np.random.normal(200000, 100000, n_samples).astype(int)


def generate_number_of_dependents(n_samples):
    return np.random.randint(0, 5, n_samples)


def generate_years_of_investing(n_samples):
    return np.random.randint(0, 40, n_samples)


def generate_synthetic_data(n_samples=10000):
    """Generates a synthetic dataset of investor profiles and their
    corresponding investment strategies."""
    np.random.seed(0)
    data = pd.DataFrame(
        {
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


if __name__ == "__main__":
    data = generate_synthetic_data()
    print(data.head())
    print(data["Investment_Strategy"].value_counts())
