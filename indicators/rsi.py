import numpy as np

class CumulatedGainsIndicator:
    """Calculates cumulated gains over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.indicator: np.ndarray = indicator
        self.timeFrame: int = timeFrame

    def getValue(self, index: int) -> float:
        """
        Calculate the cumulated gains for a given index.

        Args:
            index (int): The index to calculate gains for.

        Returns:
            float: The cumulated gains.
        """
        sumOfGains: float = 0
        for i in range(max(1, index - self.timeFrame + 1), index + 1):
            if self.indicator[i] > self.indicator[i - 1]:
                sumOfGains += self.indicator[i] - self.indicator[i - 1]
        return sumOfGains

class CumulatedLossesIndicator:
    """Calculates cumulated losses over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.indicator: np.ndarray = indicator
        self.timeFrame: int = timeFrame

    def getValue(self, index: int) -> float:
        """
        Calculate the cumulated losses for a given index.

        Args:
            index (int): The index to calculate losses for.

        Returns:
            float: The cumulated losses.
        """
        sumOfLosses: float = 0
        for i in range(max(1, index - self.timeFrame + 1), index + 1):
            if self.indicator[i] < self.indicator[i - 1]:
                sumOfLosses += self.indicator[i - 1] - self.indicator[i]
        return sumOfLosses

class AverageGainIndicator:
    """Calculates average gains over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.cumulatedGains: CumulatedGainsIndicator = CumulatedGainsIndicator(indicator, timeFrame)
        self.timeFrame: int = timeFrame

    def getValue(self, index: int) -> float:
        """
        Calculate the average gain for a given index.

        Args:
            index (int): The index to calculate average gain for.

        Returns:
            float: The average gain.
        """
        realTimeFrame: int = min(self.timeFrame, index + 1)
        return self.cumulatedGains.getValue(index) / realTimeFrame

class AverageLossIndicator:
    """Calculates average losses over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.cumulatedLosses: CumulatedLossesIndicator = CumulatedLossesIndicator(indicator, timeFrame)
        self.timeFrame: int = timeFrame

    def getValue(self, index: int) -> float:
        """
        Calculate the average loss for a given index.

        Args:
            index (int): The index to calculate average loss for.

        Returns:
            float: The average loss.
        """
        realTimeFrame: int = min(self.timeFrame, index + 1)
        return self.cumulatedLosses.getValue(index) / realTimeFrame

class RSIIndicator:
    """Calculates the Relative Strength Index (RSI) over a specified time frame."""

    def __init__(self, indicator: np.ndarray, timeFrame: int):
        self.indicator: np.ndarray = indicator
        self.timeFrame: int = timeFrame
        self.averageGainIndicator: AverageGainIndicator = AverageGainIndicator(indicator, timeFrame)
        self.averageLossIndicator: AverageLossIndicator = AverageLossIndicator(indicator, timeFrame)

    def calculate(self, index: int) -> float:
        """
        Calculate the RSI for a given index.

        Args:
            index (int): The index to calculate RSI for.

        Returns:
            float: The RSI value.
        """
        if index == 0:
            return 0

        averageLoss: float = self.averageLossIndicator.getValue(index)
        if averageLoss == 0:
            return 100

        averageGain: float = self.averageGainIndicator.getValue(index)
        relativeStrength: float = averageGain / averageLoss if averageLoss != 0 else float('inf')

        ratio: float = 100 / (1 + relativeStrength)
        return 100 - ratio
