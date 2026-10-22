import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from recsys.data.schema import AUDIO_COLUMNS


def audio_block(tracks: pd.DataFrame) -> np.ndarray:
    """Standard-scaled audio features, float32, one row per track."""
    return StandardScaler().fit_transform(tracks[AUDIO_COLUMNS]).astype(np.float32)
