from braindecode_lazy.datasets.lazy_dataset import LazyDataset
import pandas as pd
import numpy as np
import h5py
import sys

# avoid duplicates for reading file names by this ugly import
sys.path.insert(1, "/home/gemeinl/code/brainfeatures/")
from brainfeatures.data_set.tuh_abnormal import _read_all_file_names


def load_lazy_panads_h5_data(fname, start, stop):
    """load a crop from h5 file specified by fname.
    data can be n_ch x n_samp or transposed can load a crop or whole trial """
    assert fname.endswith(".h5"), "can only use h5 files"
    f = h5py.File(fname, "r")
    x_dim, y_dim = f["data"]["block0_values"].shape
    if x_dim < y_dim:
        x = f["data"]["block0_values"][:, start:stop]
    else:
        x = f["data"]["block0_values"][start:stop, :].T
    f.close()
    x = np.array(x, dtype=np.float32)
    return x


class TuhLazy(LazyDataset):
    """Tuh lazy data set. """
    def __init__(self, data_folder, n_recordings=None, target="pathological"):
        self.task = target
        assert data_folder.endswith("/"), "data_folder has to end with '/'"
        self.files = _read_all_file_names(data_folder, ".h5", key="time")

        if n_recordings is not None:
            self.files = self.files[:n_recordings]

        self.X, self.y = self.load(self.files)

    def load(self, files):
        X, y = [], []
        for file_ in files:
            # pandas read is slow
            # however, this is only called once upon creation of the data set
            info_df = pd.read_hdf(file_, key="info")
            assert len(info_df) == 1, "too many rows in info df"
            info = info_df.iloc[-1].to_dict()
            y_ = info[self.task]
            if self.task == "gender":
                y_ = 0 if info["gender"] == "M" else 1
            y.append(y_)
            n_samples = int(info["n_samples"])
            X.append(np.empty(shape=(1, n_samples)))

        if self.task == "age":
            y = np.array(y).astype(np.float32)
        else:
            assert self.task in ["pathological", "gender"], "unknown task"
            y = np.array(y).astype(np.int64)
        return X, y

    def load_lazy(self, fname, start, stop):
        return load_lazy_panads_h5_data(fname, start, stop)


class TuhLazySubset(LazyDataset):
    """ A subset of a tuh lazy data set based on indices."""
    def __init__(self, dataset, indices):
        self.task = dataset.task
        self.files = np.array([dataset.files[i] for i in indices])

        self.X, self.y = dataset.load(self.files)

    def load_lazy(self, fname, start, stop):
        return load_lazy_panads_h5_data(fname, start, stop)
