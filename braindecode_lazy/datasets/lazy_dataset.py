from abc import ABC, abstractmethod


class LazyDataset(ABC):
    """ Class implementing an abstract lazy data set. Custom lazy data sets
    can inherit from this class. they have to override files, X and y as well
    as the load_lazy function.
    """
    def __init__(self):
        self.files = "Not implemented: a list of all file names in the dataset"
        self.X = "Not implemented: a list of empty ndarrays with second " \
                 "dimension equal to number of samples of each example." \
                 "used to create crops"
        self.y = "Not implemented: a list of all targets in the dataset"

    @abstractmethod
    def load_lazy(self, fname, start, stop):
        """ Loading procedure that gets a filename, start and stop indices and
        returns a signal trial / crop. set start and stop to None to load a trial"""
        raise NotImplementedError

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ Returns a two-tuple of example, label """
        # when cropping idx is a 3-tuple holding trial id, start ind and stop ind
        try:
            idx, start, stop = idx
        except (TypeError, ValueError):
            start = 0
            stop = None
        fname = self.files[idx]
        x = self.load_lazy(fname, start, stop)

        if x.ndim == 2:
            x = x[:, :, None]
        return x, self.y[idx]
