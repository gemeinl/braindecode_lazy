import numpy as np
import h5py
import json
import sys

# avoid duplicates for reading file names by this ugly import
sys.path.insert(1, "/home/gemeinl/code/brainfeatures/")
from brainfeatures.data_set.tuh_abnormal import _read_all_file_names


class Tuh(object):
    def __init__(self, data_folder, n_recordings=None, target="pathological",
                 extension=".npy"):
        self.task = target
        self.extension = extension
        self.files = _read_all_file_names(data_folder, extension=extension,
                                          key="time")
        if n_recordings is not None:
            self.files = self.files[:n_recordings]

        self.X, self.y = self.load(self.files)

    def load(self, files):
        X, y = [], []
        for file_ in files:
            if self.extension == ".npy":
                x = np.load(file_).astype(np.float32)
            else:
                assert self.extension == ".h5", "unknown file format"
                f = h5py.File(file_, "r")
                x = f["signals"][:]
                f.close()
            xdim, ydim = x.shape
            if xdim > ydim:
                x = x.T
            x = np.array(x).astype(np.float32)
            X.append(x)
            json_file = file_.replace(self.extension, ".json")
            with open(json_file, "r") as f:
                info = json.load(f)
            y_ = info[self.task]
            if self.task == "gender":
                y_ = 0 if info["gender"] == "M" else 1
            y.append(y_)

        if self.task == "age":
            y = np.array(y).astype(np.float32)
        else:
            assert self.task in ["pathological", "gender"], "unknown task"
            y = np.array(y).astype(np.int64)
        return X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TuhSubset(object):
    def __init__(self, dataset, indices):
        self.X = [dataset.X[i] for i in indices]
        self.y = np.array([dataset.y[i] for i in indices])
        self.files = np.array([dataset.files[i] for i in indices])
