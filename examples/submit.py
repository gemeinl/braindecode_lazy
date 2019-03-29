import pandas as pd
import numpy as np
import os

from utils import parse_submit_args

# TODO: seed solution is really ugly!


def dict_to_cmd_args(d):
    cmd_args = []
    for key, value in sorted(d.items()):
        if pd.isna(value):  # or value == "None" or value is None:
            continue
        cmd_args.append("--"+key+"="+str(value))
    return " ".join(cmd_args)


def read_job_id_from_job_name(job_name):
    # find slurm job id based on given job name and read it
    job_id = os.popen('squeue --noheader --format %i --name {}'
                      .format(job_name)).read()
    # since they are all array jobs, only take the job id not the array id
    dependency_job_id = job_id.split("_")[0]
    return dependency_job_id


def main(configs_file, conda_env_name, python_file, queue, python_path, start,
         stop):
    # create jobs file text in this script from which a temporary file will
    # be created
    job_file = (
        "#!/bin/bash\n"
        "# redirect the output/error to some files\n"
        "#SBATCH -o /home/gemeinl/out/%A-%a.o\n"
        "#SBATCH -e /home/gemeinl/error/%A-%a.e\n"
        "export PYTHONPATH={}\n"
        "source {}\n"
        "python {} {}\n")

    # load all the configs to be run
    configs_df = pd.read_csv(configs_file, index_col=0, dtype=str)
    # remove empy and comment lines starting with '#'
    configs_df.drop([index for index in configs_df.index if
                     index is np.nan or index.startswith("#")], inplace=True)
    # activate conda env
    virtual_env = ('/home/gemeinl/anaconda3/bin/activate && '
                   'conda activate {}'.format(conda_env_name))
    # create a tmp job file to be run
    script_name = "/home/gemeinl/jobs/slurm/run_tmp.sh"
    batch_submit = "sbatch -p {} -c {}"
    if start is not None and stop is not None:
        batch_submit += " --array={}-{}"
    batch_submit += " --job-name={} {}"
    # add n_parallel? 0-5:5
    # add time limit? --time=12:00:00"

    # sbatch -p meta_gpu-ti -w metagpub -c 4 jobs/slurmbjob.pbs
    for setting_i, (setting_name, setting) in enumerate(configs_df.iterrows()):
        config = configs_df.loc[setting_name].to_dict()
        # n = int(config["n_folds"])
        if "num_workers" in config:
            num_workers = int(config["num_workers"])
        else:
            num_workers = 1
        cmd_args = dict_to_cmd_args(config)
        curr_job_file = job_file.format(python_path, virtual_env, python_file,
                                        cmd_args)
        with open(script_name, "w") as f:
            f.writelines(curr_job_file)

        command = batch_submit.format(
            queue, num_workers, start, stop, setting_name, script_name)
        print(command)
        os.system(command)


if __name__ == '__main__':
    kwargs = parse_submit_args()
    print(kwargs)
    main(**kwargs)
