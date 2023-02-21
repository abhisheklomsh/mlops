import math

import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    expected_first_prediction_value = 1.0
    expected_no_predictions = 262
    abs_tolerance = 100

    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert len(predictions) == expected_no_predictions
    assert isinstance(predictions[0], np.float64)
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=abs_tolerance)

    assert result.get("errors") is None
