from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from numpy.random import RandomState
import numpy as np

from braindecode.datautil.iterators import _compute_start_stop_block_inds, \
    get_balanced_batches


def custom_collate(batch):
    """ taken and adapted from pytorch
    https://pytorch.org/docs/0.4.1/_modules/torch/utils/data/dataloader.html
    """
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy':
        return np.stack([b for b in batch], 0)

    elif isinstance(batch[0], tuple):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]


class LoadCropsFromTrialsIterator(object):
    """ This is basically the same code as CropsFromTrialsIterator from
    braindecode. However, it is adapted to work with lazy datasets. It uses
    pytorch data loaders to load recordings from hdd with multiple threads
    when the data is actually needed. Reduces overall RAM requirements.
    """
    def __init__(self, input_time_length, n_preds_per_input, batch_size,
                 seed=328774, num_workers=0, collate_fn=custom_collate):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = RandomState(self.seed)
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def reset_rng(self):
        self.rng = RandomState(self.seed)

    def get_n_updates_per_epoch(self, dataset, shuffle):
        bach_sampler = mySampler(dataset, self.input_time_length,
                                 self.batch_size, shuffle,
                                 self.n_preds_per_input, self.rng)
        return len(bach_sampler)

    def get_batches(self, dataset, shuffle):
        bach_sampler = mySampler(dataset, self.input_time_length,
                                 self.batch_size, shuffle,
                                 self.n_preds_per_input, self.rng)
        data_loader = DataLoader(dataset=dataset, batch_sampler=bach_sampler,
                                 num_workers=self.num_workers, pin_memory=False,
                                 collate_fn=self.collate_fn)
        return data_loader


class mySampler(Sampler):
    def __init__(self, dataset, input_time_length, batch_size, shuffle, n_preds_per_input, rng):
        # start always at first predictable sample, so
        # start at end of receptive field
        n_receptive_field = input_time_length - n_preds_per_input + 1
        i_trial_starts = [n_receptive_field - 1] * len(dataset.X)
        i_trial_stops = [trial.shape[1] for trial in dataset.X]

        # Check whether input lengths ok
        input_lens = i_trial_stops
        for i_trial, input_len in enumerate(input_lens):
            assert input_len >= input_time_length, (
                "Input length {:d} of trial {:d} is smaller than the "
                "input time length {:d}".format(input_len, i_trial,
                                                input_time_length))

        start_stop_blocks_per_trial = _compute_start_stop_block_inds(
            i_trial_starts, i_trial_stops, input_time_length,
            n_preds_per_input, check_preds_smaller_trial_len=True)
        for i_trial, trial_blocks in enumerate(start_stop_blocks_per_trial):
            assert trial_blocks[0][0] == 0
            assert trial_blocks[-1][1] == i_trial_stops[i_trial]

        i_trial_start_stop_block = np.array([
            (i_trial, start, stop) for i_trial, block in
            enumerate(start_stop_blocks_per_trial) for (start, stop) in block])

        batches = get_balanced_batches(
            n_trials=len(i_trial_start_stop_block), rng=rng,
            shuffle=shuffle, batch_size=batch_size)

        self.batchs = [i_trial_start_stop_block[batch_ind] for batch_ind in batches]

    def __iter__(self):
        for batch in self.batchs:
            yield batch

    def __len__(self):
        len(self.batchs)
