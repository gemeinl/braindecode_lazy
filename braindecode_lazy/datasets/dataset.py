from abc import ABC


class Dataset(ABC):
    def __init__(self):
        self.file_paths = "Not implemented: a list of all file paths"
        self.X = ("Not implemented: a list of empty ndarrays with number of "
                  "samples as second dimension")
        self.y = "Not implemented: a list of all targets"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
