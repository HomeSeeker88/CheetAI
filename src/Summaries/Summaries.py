from typing import Optional

import numpy as np
from pydantic import BaseModel


class Summary(BaseModel):
    precision: Optional[np.float64]
    recall: Optional[np.float64]
    accuracy: Optional[np.float64]

    class Config:
        from_attributes = True

class LinearRegressionSummary(Summary):
    """
    rmse: Root Mean Squared Error
    mse: Mean Squared Error
    mae: Mean Absolute Error
    """
    rmse: np.float64
    mse: np.float64
    mae: np.float64
    r_squared: np.float64