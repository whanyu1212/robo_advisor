import pandera as pa
from pandera import Check, Column, DataFrameSchema

out_schema = DataFrameSchema(
    {
        "id": Column(pa.Int, Check.greater_than_or_equal_to(0)),
    }
)
