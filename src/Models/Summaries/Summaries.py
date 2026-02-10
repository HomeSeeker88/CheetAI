from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class Summary(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None

    class Config:
        from_attributes = True

class LinearRegressionSummary(Summary):
    """
    rmse: Root Mean Squared Error
    mse: Mean Squared Error
    mae: Mean Absolute Error
    r_squared: R Squared
    error_normality: error normality
    """
    rmse: float
    mse: float
    mae: float
    r_squared: float
    error_normality: bool

    model_config = ConfigDict(arbitrary_types_allowed = True, from_attributes = True)

