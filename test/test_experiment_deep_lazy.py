from examples.tuh_auto_diag_lazy_without_memleak import setup_exp
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

    exp = setup_exp(**kwargs)
    # for testing purposes, reset rng after each
    exp.iterator.reset_rng_after_each_batch = True
    exp.run()

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([1.3472894430160522, 133.92396545410156, 15.145020484924316]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([1.7500500679016113, 0.0, 0.0]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.4353356481481482, 0.25, 0.26443287037037033]),
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

    # for testing purposes, reset rng after each
    exp = setup_exp(**kwargs)
    exp.iterator.reset_rng_after_each_batch = True
    exp.run()

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([0.9724488258361816, 621.3842163085938, 233.0751190185547]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([3.381948232650757, 0.0, 0.0]),
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
