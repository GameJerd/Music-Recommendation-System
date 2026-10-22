import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer


def genre_block(tracks: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    """Multi-hot of artist genres (sparse; ~3k columns). Tracks with no genres get an empty row."""
    mlb = MultiLabelBinarizer(sparse_output=True)
    m = mlb.fit_transform(tracks["genres"]).astype(np.float32).tocsr()
    return m, list(mlb.classes_)


def decade_block(tracks: pd.DataFrame) -> tuple[sparse.csr_matrix, list[int]]:
    decades = (tracks["year"].to_numpy() // 10) * 10
    labels, codes = np.unique(decades, return_inverse=True)
    m = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(labels)),
    )
    return m, labels.tolist()
