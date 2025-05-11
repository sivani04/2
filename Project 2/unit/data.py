from collections import namedtuple
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from util import logutil

logger = logutil.logger_run

Data = namedtuple('Data', 'X_embedding X_text y_type y_index')


def make_data(embedding: np.ndarray,
              df: pd.DataFrame,
              types: List[str]) -> Data:
    X_embedding = embedding
    X_text = df[config.TEXT_COL].to_numpy()
    y_type = df['y'] if 'y' in df else None
    y_index = [types.index(y) for y in y_type] if y_type else None
    return Data(X_embedding, X_text, y_type, y_index)


def split_data(embedding: np.ndarray,
               df: pd.DataFrame) -> Tuple:
    good_types = df['y'].value_counts()[df['y'].value_counts() >= 3].index

    if len(good_types) < 1:
        logger.info("None of the class have more than 3 records: Skipping ...")
        return None, None, good_types

    X_embedding = embedding[df['y'].isin(good_types)]
    X_text = df[config.TEXT_COL][df['y'].isin(good_types)]
    y_type = df['y'][df['y'].isin(good_types)]
    types = np.unique(y_type).tolist()
    y_index = [types.index(y) for y in y_type]

    (X_embedding_train, X_embedding_test, X_text_train, X_text_test,
     y_type_train, y_type_test, y_index_train, y_index_test) = train_test_split(X_embedding,
                                                                                X_text,
                                                                                y_type,
                                                                                y_index,
                                                                                test_size=config.TEST_SIZE,
                                                                                random_state=config.SEED,
                                                                                stratify=y_type)

    train_data = Data(X_embedding_train, X_text_train, y_type_train, y_index_train)
    test_data = Data(X_embedding_test, X_text_test, y_type_test, y_index_test)
    return train_data, test_data, types
