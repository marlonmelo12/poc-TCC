"""
Data loading and preprocessing module for bioinformatics benchmark datasets.
Handles:
- Colon Cancer (Alon et al., 1999) - 62 samples, 2000 features
- Prostate Tumor (Singh et al., 2002) - 102 samples, 6033 features
- Ovarian Cancer (Petricoin et al., 2002 / Kent Ridge) - 253 samples, 15154 features
"""

import os
import io
import lzma
import tarfile
import urllib.request
from typing import Tuple
import numpy as np
import pandas as pd

try:
    import rdata
except ImportError:
    rdata = None


DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

URLS = {
    "colon": "https://openml.org/data/v1/download/22112148/Colon.arff",
    "ovarian": "https://openml.org/data/v1/download/22112159/Ovarian.arff",
    "prostate_cran": "https://cran.r-project.org/src/contrib/sda_1.3.9.tar.gz",
}


def _download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


def _parse_arff(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="latin1") as f:
        lines = f.read().splitlines()

    attr_names = []
    data_idx = -1
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.lower().startswith("@attribute"):
            parts = line_s.split()
            attr_names.append(parts[1].strip("'\""))
        elif line_s.lower().startswith("@data"):
            data_idx = i + 1
            break

    if data_idx == -1:
        raise ValueError(f"Could not find @data marker in {file_path}")

    data_str = "\n".join(lines[data_idx:])
    df = pd.read_csv(io.StringIO(data_str), names=attr_names, comment="%")
    return df


def load_colon(force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads Colon Cancer dataset (62 samples, 2000 features).
    Classes: 0 (tumor), 1 (normal).
    """
    processed_file = os.path.join(DATA_PROCESSED_DIR, "colon.parquet")
    raw_file = os.path.join(DATA_RAW_DIR, "Colon.arff")

    if not force_download and os.path.exists(processed_file):
        df = pd.read_parquet(processed_file)
    else:
        _download_file(URLS["colon"], raw_file)
        df = _parse_arff(raw_file)
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        df.to_parquet(processed_file, index=False)

    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X = df[feature_cols].values.astype(np.float64)
    # Map classes to binary 0/1: -1 (tumor) -> 0, 1 (normal) -> 1 or string labels
    y_raw = df[target_col].values
    unique_vals = np.unique(y_raw)
    y = np.where(y_raw == unique_vals[0], 0, 1).astype(np.int64)

    assert X.shape == (62, 2000), f"Colon shape mismatch: {X.shape} != (62, 2000)"
    assert len(np.unique(y)) == 2, f"Target classes mismatch: {np.unique(y)}"
    assert not np.isnan(X).any(), "Colon contains NaN values"

    return X, y, list(feature_cols)


def load_ovarian(force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads Ovarian Cancer dataset (253 samples, 15154 features).
    Classes: 0 (Cancer), 1 (Normal).
    """
    processed_file = os.path.join(DATA_PROCESSED_DIR, "ovarian.parquet")
    raw_file = os.path.join(DATA_RAW_DIR, "Ovarian.arff")

    if not force_download and os.path.exists(processed_file):
        df = pd.read_parquet(processed_file)
    else:
        _download_file(URLS["ovarian"], raw_file)
        df = _parse_arff(raw_file)
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        df.to_parquet(processed_file, index=False)

    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X = df[feature_cols].values.astype(np.float64)
    y_raw = df[target_col].values
    # Classes: 'Cancer' -> 0, 'Normal' -> 1
    y = np.where(y_raw == "Cancer", 0, 1).astype(np.int64)

    assert X.shape == (253, 15154), f"Ovarian shape mismatch: {X.shape} != (253, 15154)"
    assert len(np.unique(y)) == 2, f"Target classes mismatch: {np.unique(y)}"
    assert not np.isnan(X).any(), "Ovarian contains NaN values"

    return X, y, list(feature_cols)


def load_prostate(force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads Prostate Tumor dataset (Singh et al., 2002: 102 samples, 6033 features).
    Classes: 0 (cancer/tumor), 1 (normal/healthy).
    """
    processed_file = os.path.join(DATA_PROCESSED_DIR, "prostate.parquet")
    raw_tar = os.path.join(DATA_RAW_DIR, "sda_1.3.9.tar.gz")
    raw_rda = os.path.join(DATA_RAW_DIR, "singh2002.rda")

    if not force_download and os.path.exists(processed_file):
        df = pd.read_parquet(processed_file)
        target_col = df.columns[-1]
        feature_cols = df.columns[:-1]
        X = df[feature_cols].values.astype(np.float64)
        y = df[target_col].values.astype(np.int64)
        return X, y, list(feature_cols)

    _download_file(URLS["prostate_cran"], raw_tar)

    if not os.path.exists(raw_rda):
        with tarfile.open(raw_tar, mode="r:gz") as tar:
            member = tar.getmember("sda/data/singh2002.rda")
            with tar.extractfile(member) as src, open(raw_rda, "wb") as dst:
                dst.write(src.read())

    if rdata is None:
        raise ImportError("Package 'rdata' is required to parse singh2002.rda")

    parsed = rdata.parser.parse_file(raw_rda)
    converted = rdata.conversion.convert(parsed)
    singh_data = converted["singh2002"]

    # singh2002 has 'x' (102 x 6033 matrix) and 'y' (factor with 102 labels)
    X = np.asarray(singh_data["x"], dtype=np.float64)
    y_raw = np.asarray(singh_data["y"])
    
    unique_vals = np.unique(y_raw)
    # Map cancer -> 0, healthy/normal -> 1
    y = np.where(y_raw == unique_vals[0], 0, 1).astype(np.int64)
    feature_cols = [f"gene_{i+1}" for i in range(X.shape[1])]

    assert X.shape == (102, 6033), f"Prostate shape mismatch: {X.shape} != (102, 6033)"
    assert len(np.unique(y)) == 2, f"Target classes mismatch: {np.unique(y)}"
    assert not np.isnan(X).any(), "Prostate contains NaN values"

    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_parquet(processed_file, index=False)

    return X, y, feature_cols


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, list]:
    name_clean = name.strip().lower()
    if "colon" in name_clean:
        return load_colon()
    elif "ovarian" in name_clean:
        return load_ovarian()
    elif "prostate" in name_clean:
        return load_prostate()
    else:
        raise ValueError(f"Unknown dataset name: {name}")
