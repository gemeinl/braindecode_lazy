from examples.tuh_auto_diag_lazy_without_memleak import run_exp
import numpy as np

# TODO: implement age tests

DATA_FOLDER = "/home/gemeinl/code/braindecode_lazy/test/files/"


def test_run_shallow_pathological():
    kwargs = {
        "train_folder": DATA_FOLDER,
        "batch_size": 64,
        "max_epochs": 2,
        "cuda": False,
        "seed": 0,
        "num_workers": 0,
        "task": "pathological",
        "n_recordings": None,
        "shuffle_folds": False,
        "n_folds": 5,
        "eval_folder": None,
        "lazy_loading": True,
        "result_folder": None,

        # model specific
        "model_name": "shallow",
        "n_start_chans": 40,
        "n_chans": 21,
        "n_chan_factor": None,
        "final_conv_length": 35,
        "model_constraint": None,
        "init_lr": 0.0625 * 0.01,
        "weight_decay": 0,
        "input_time_length": 6000,
        "stride_before_pool": None,
    }

    exp = run_exp(**kwargs)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([5.6385579109191895, 0.9429725408554077, 0.9677126407623291]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([15.66110610961914, 5.81709098815918, 6.502216339111328]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.7478788019287834, 0.4502619621661721, 0.44074554896142437]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_sample_misclass),
        np.array([1.0, 1.0, 1.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_misclass),
        np.array([0.75, 0.5, 0.5]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_misclass),
        np.array([1.0, 1.0, 1.0]),
        atol=1e-3, rtol=1e-3)


def test_run_shallow_gender():
    kwargs = {
        "train_folder": DATA_FOLDER,
        "batch_size": 64,
        "max_epochs": 2,
        "cuda": False,
        "seed": 0,
        "num_workers": 0,
        "task": "gender",
        "n_recordings": None,
        "shuffle_folds": False,
        "n_folds": 5,
        "eval_folder": None,
        "lazy_loading": True,
        "result_folder": None,

        # model specific
        "model_name": "shallow",
        "n_start_chans": 40,
        "n_chans": 21,
        "n_chan_factor": None,
        "final_conv_length": 35,
        "model_constraint": None,
        "init_lr": 0.0625 * 0.01,
        "weight_decay": 0,
        "input_time_length": 6000,
        "stride_before_pool": None,
    }

    exp = run_exp(**kwargs)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([4.7572245597839355, 2.673903465270996, 1.164989948272705]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([3.700687329910579e-07, 2.1586083676083945e-07, 1.9874266854458256e-06]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.7478788019287834, 0.5813125927299703, 0.3513654488130564]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_sample_misclass),
        np.array([0.0, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_misclass),
        np.array([0.75, 0.75, 0.25]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_misclass),
        np.array([0.0, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)
