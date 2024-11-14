import numpy as np

class SMAIndicator:
    """Calculates the Simple Moving Average (SMA) over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.indicator: np.ndarray = indicator
        self.timeFrame: int = timeFrame

    def calculate(self, index: int) -> float:
        """
        Calculate the SMA for a given index.

        Args:
            index (int): The index to calculate SMA for.

        Returns:
            float: The SMA value.
        """
        start_index: int = max(0, index - self.timeFrame + 1)
        window_values: np.ndarray = self.indicator[start_index:index + 1]
        return float(np.mean(window_values))
