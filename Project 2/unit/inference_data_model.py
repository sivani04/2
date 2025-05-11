import numpy as np
import pandas as pd

from innso_ticket import config
from innso_ticket.util import logutil

logger = logutil.logger_run


class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        X_DL = df[config.TEXT_COL]
        X_DL = X_DL.to_numpy()
        y = df.y.to_numpy()
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 1:
            logger.info("None of the class have more than 3 records: Skipping ...")
            self.X_test = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        X_DL_good = X_DL[y_series.isin(good_y_value)]
        y_filtered, indexed_y_filtered = np.unique(y_good, return_inverse=True)
        self.X_test = X_good
        self.y_test = y_good
        self.X_DL_test = X_DL_good
        self.y_DL_test = indexed_y_filtered

        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X
