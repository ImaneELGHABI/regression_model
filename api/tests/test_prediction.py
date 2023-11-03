import math
import json
import requests
import numpy as np
import pandas as pd
from regression_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 113422
    expected_no_predictions = 1449
    sample_input_data = sample_input_data.to_json(orient="records").replace('"', '\\"')
    # When
    result = requests.post(
        "http://127.0.0.1:8000/predict",
        data=f'{{"input_data": "{sample_input_data}"}}',
        headers={"Content-Type": "application/json"},
        ).json()
    # Then
    predictions = [ np.float64(i) for i in result.get("predictions")]
    print(predictions)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
