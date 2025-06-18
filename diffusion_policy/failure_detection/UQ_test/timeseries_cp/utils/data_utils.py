from enum import Enum
from typing import Dict
import numpy as np

from diffusion_policy.failure_detection.UQ_test.timeseries_cp.problems.stochastic_processes import AbstractStochasticProcess


def generate_and_split_data(
    stochastic_process: AbstractStochasticProcess,
    rng: np.random.Generator,
    length: int,
    train_size: int,
    calibration_size: int,
    test_size: int = 1,
) -> Dict:
    """Generates and splits data into training, calibration, and testing.

    Args:
        stochastic_process: An AbstractStochasticProcess object.
        rng: An np.random.Generator object.
        length: The length of each trajectory in the data.
        train_size: Batch size of the training set.
        calibration_size: Batch size of the calibration set.
        test_size (optional): Batch size of the test set. Defaults to 1.

    Returns:
        A dict with keys: "train", "calibration", and "test", where each value is np.ndarray.
    """
    train_data = stochastic_process.generate_data(train_size, length, rng)
    calibration_data = stochastic_process.generate_data(calibration_size, length, rng)
    test_data = stochastic_process.generate_data(test_size, length, rng)
    data = {"train": train_data, "calibration": calibration_data, "test": test_data}
    return data


class RegressionType(Enum):
    Mean = 1
    ConstantMean = 2


def regress(training_data: np.ndarray, regression_type: RegressionType) -> np.ndarray:
    """Regresses training data to a single trajectory.

    Args:
        training_data: An np.ndarray object of shape (batch_size, length).
        regression_type: A regression type.

    Returns:
        A np.ndarray object of shape (1, length).

    Raises:
        NotImplementedError if unknown regression_type is given.
    """
    # import pdb; pdb.set_trace()
    print(f"regression_type: {regression_type}")
    if regression_type == RegressionType.Mean:
        return np.mean(training_data, axis=0, keepdims=True)
    elif regression_type == RegressionType.ConstantMean:
        const_mean = np.mean(training_data)
        return const_mean * np.ones((1, training_data.shape[1]))
    else:
        raise (NotImplementedError)