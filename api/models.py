import json
import pandas as pd

from pydantic import BaseModel, validator
from regression_model.processing.validation import validate_inputs


class PredictionRequest(BaseModel):
    """
    Expected prediction request.
    """

    input_data: str

    @validator("input_data")
    def validate_input_data(cls, v):
        v = pd.DataFrame(json.loads(v))
        validated, errors = validate_inputs(input_data=v)
        if errors:
            raise ValueError(errors)
        return validated
