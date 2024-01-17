age_threshold = 70
net_worth_threshold = 1_000_000_000

strategy_conditions = [
    {
        "strategy": "Aggressive",
        "conditions": [
            lambda row: row["Risk_Tolerance"] == "High",
            lambda row: row["Financial_Knowledge"] == "High",
        ],
    },
    {
        "strategy": "Conservative",
        "conditions": [
            lambda row: row["Risk_Tolerance"] == "Low",
            lambda row: row["Time_Horizon"] == "Short-term",
        ],
    },
    {
        "strategy": "Growth-oriented",
        "conditions": [
            lambda row: row["Income_Level"] == "High",
            lambda row: row["Credit_Score"] > 750,
        ],
    },
    {
        "strategy": "Growth",
        "conditions": [
            lambda row: row["Age"] > age_threshold,
            lambda row: row["Net_Worth"] > net_worth_threshold,
        ],
    },
    {
        "strategy": "Income",
        "conditions": [
            lambda row: row["Age"] <= age_threshold,
            lambda row: row["Net_Worth"] > net_worth_threshold,
        ],
    },
    {
        "strategy": "Preservation",
        "conditions": [
            lambda row: row["Age"] > age_threshold,
            lambda row: row["Net_Worth"] <= net_worth_threshold,
        ],
    },
]
