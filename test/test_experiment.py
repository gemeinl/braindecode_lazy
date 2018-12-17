from examples.tuh_auto_diag_lazy import run_exp
import numpy as np

# TODO: implement age tests

DATA_FOLDER = "/home/gemeinl/code/braindecode_lazy/test/files/"


def test_run_deep_pathological():
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
        "model_name": "deep",
        "n_start_chans": 25,
        "n_chans": 21,
        "n_chan_factor": 2,
        "final_conv_length": 1,
        "model_constraint": None,
        "init_lr": 0.01,
        "weight_decay": 0.5 * 0.001,
        "input_time_length": 6000,
        "stride_before_pool": True,
    }

    exp = run_exp(**kwargs)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([1.3472893238067627, 143.13360595703125, 20.755834579467773]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([1.7500501871109009, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.4353356481481482, 0.25, 0.2620949074074074]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_sample_misclass),
        np.array([0.43921296296296297, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_misclass),
        np.array([0.25, 0.25, 0.25]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_misclass),
        np.array([0.0, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)


def test_run_deep_gender():
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
        "model_name": "deep",
        "n_start_chans": 25,
        "n_chans": 21,
        "n_chan_factor": 2,
        "final_conv_length": 1,
        "model_constraint": None,
        "init_lr": 0.01,
        "weight_decay": 0.5 * 0.001,
        "input_time_length": 6000,
        "stride_before_pool": True,
    }

    exp = run_exp(**kwargs)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([0.9724490642547607, 658.634521484375, 228.96507263183594]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([3.3819427490234375, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.4196643518518518, 0.75, 0.75]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_sample_misclass),
        np.array([0.560787037037037, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_misclass),
        np.array([0.25, 0.75, 0.75]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_misclass),
        np.array([1.0, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)


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
