from examples.tuh_auto_diag import setup_exp
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
        "lazy_loading": False,
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

    exp = setup_exp(**kwargs)
    exp.run()

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([5.638560771942139, 0.9429671168327332, 0.9720389246940613]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([15.661099433898926, 5.816369533538818, 6.5150346755981445]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.7478788019287834, 0.450447422106825, 0.44304061572700293]),
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
        "lazy_loading": False,
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

    exp = setup_exp(**kwargs)
    exp.run()

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_loss),
        np.array([4.757225513458252, 2.666571617126465, 1.1629936695098877]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.test_loss),
        np.array([3.6769267808267614e-07, 2.169014123865054e-07, 2.007234570555738e-06]),
        atol=1e-3, rtol=1e-3)

    np.testing.assert_allclose(
        np.array(exp.epochs_df.train_sample_misclass),
        np.array([0.7478788019287834, 0.5806750741839762, 0.35107566765578635]),
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
