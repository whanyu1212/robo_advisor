from pandera import Check, Column, DataFrameSchema

synthetic_data_schema = DataFrameSchema(
    {
        "User_ID": Column(str, Check.str_length(10, 10), nullable=True),
        "Age": Column(
            int, Check.greater_than_or_equal_to(18), Check.less_than_or_equal_to(100)
        ),
        "Income_Level": Column(str, Check.isin(["Low", "Medium", "High"])),
        "Credit_Score": Column(
            int, Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1000)
        ),
        "Investment_Experience": Column(
            str, Check.isin(["Novice", "Intermediate", "Experienced"])
        ),
        "Financial_Knowledge": Column(str, Check.isin(["Low", "Medium", "High"])),
        "Risk_Tolerance": Column(str, Check.isin(["Low", "Medium", "High"])),
        "Investment_Goals": Column(
            str, Check.isin(["Preservation", "Income", "Growth", "Speculation"])
        ),
        "Time_Horizon": Column(
            str, Check.isin(["Short-term", "Medium-term", "Long-term"])
        ),
        "Total_Assets": Column(float, Check.greater_than_or_equal_to(0)),
        "Total_Liabilities": Column(float, Check.greater_than_or_equal_to(0)),
        "Number_of_Dependents": Column(int, Check.greater_than_or_equal_to(0)),
        "Years_of_Investing": Column(int, Check.greater_than_or_equal_to(0)),
        "Net_Worth": Column(float, Check.greater_than_or_equal_to(0)),
        "Investment_Strategy": Column(
            str,
            Check.isin(
                [
                    "Aggressive",
                    "Conservative",
                    "Growth-oriented",
                    "Growth",
                    "Income",
                    "Preservation",
                    "Balanced",
                ]
            ),
        ),
    }
)
