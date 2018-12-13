from braindecodelazy.datasets.lazy_dataset import LazyDataset
import numpy as np
import h5py
import json
import sys

# avoid duplicates for reading file names by this ugly import
sys.path.insert(1, "/home/gemeinl/code/brainfeatures/")
from brainfeatures.data_set.tuh_abnormal import _read_all_file_names
# TODO: use this?
# from brainfeaturedecode.utils.file_util import h5_load, numpy_load

# TODO: standardize age


def load_lazy(fname, start, stop):
    """load a crop from file specified by fname.
    can be either numpy or h5 file. data can be n_ch x n_samp or transposed
    can load a crop or whole trial """
    if fname.endswith(".npy"):
        x = np.load(fname)
        x_dim, y_dim = x.shape
        # assume number of channels is always smaller number of samples
        if x_dim < y_dim:
            x = x[:, start:stop]
        else:
            x = x[start:stop, :].T
    else:
        assert fname.endswith(".h5"), "unknown extension"
        f = h5py.File(fname, "r")
        x_dim, y_dim = f["signals"].shape
        if x_dim < y_dim:
            x = f["signals"][:, start:stop]
        else:
            x = f["signals"][start:stop, :].T
        f.close()
    x = x.astype(np.float32)
    return x


class TuhLazy(LazyDataset):
    """Tuh lazy data set.
    """
    def __init__(self, data_folder, n_recordings=None, target="pathological",
                 extension=".h5"):
        self.task = target
        self.extension = extension
        self.files = _read_all_file_names(data_folder, extension, key="time")

        if n_recordings is not None:
            self.files = self.files[:n_recordings]

        self.X, self.y = self.load(self.files)

    def load(self, files):
        X, y = [], []
        for file_ in files:
            json_file = file_.replace(self.extension, ".json")
            with open(json_file, "r") as f:
                info = json.load(f)
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
        return load_lazy(fname, start, stop)


class TuhLazySubset(LazyDataset):
    """ A subset of a tuh lazy data set based on indices."""
    def __init__(self, dataset, indices):
        self.task = dataset.task
        self.files = np.array([dataset.files[i] for i in indices])

        self.X, self.y = dataset.load(self.files)

    def load_lazy(self, fname, start, stop):
        return load_lazy(fname, start, stop)
