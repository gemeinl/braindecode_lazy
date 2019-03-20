import pandas as pd
import os

from utils import parse_submit_args


# TODO: use .txt file instead of .csv file?


# convert dictionay to cmd args in the form "--key value"
def dict_to_cmd_args(d):
    s = []
    for key, value in d.items():
        # nan means empty entry in csv file
        # nan has type float, everything else is string
        # some cmd args should be '--test' and '--no-test'
        if type(value) is float:
            s.append("--" + key)
        else:
            s.append("--"+key+" "+str(value))
    return " ".join(s)


def read_job_id_from_job_name(job_name):
    # find slurm job id based on given job name and read it
    job_id = os.popen('squeue --noheader --format %i --name {}'
                      .format(job_name)).read()
    # since they are all array jobs, only take the job id not the array id
    dependency_job_id = job_id.split("_")[0]
    return dependency_job_id


def main(configs_file, conda_env_name, python_file, queue, python_path):
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
    configs_df = pd.read_csv(configs_file, index_col=0)
    # activate conda env
    virtual_env = 'conda activate {}'.format(conda_env_name)
    # create a tmp job file to be run
    script_name = "/home/gemeinl/jobs/slurm/run_tmp.sh"
    batch_submit = "sbatch -p {} -c {} --array={}-{} --job-name={} {}"
    # --time=12:00:00"

    # sbatch -p meta_gpu-ti -w metagpub -c 4 jobs/slurmbjob.pbs

    setting_names = list(configs_df.columns)
    # loop through all the configs
    for setting_i, setting in enumerate(configs_df):
        config = configs_df[setting].to_dict()
        # TODO: what if n_folds not in config?
        n = int(config["n_folds"])
        num_workers = int(config["num_workers"])
        # create a tmp job file / job for every repetition / fold
        for j in range(n):
            config["seed"] = j
            cmd_args = dict_to_cmd_args(config)
            curr_job_file = job_file.format(python_path, virtual_env,
                                            python_file, cmd_args)

            # write tmp job file and submit it to slurm
            with open(script_name, "w") as f:
                f.writelines(curr_job_file)

            print(batch_submit.format(
                queue, num_workers, j, j, setting_names[setting_i]+str(j),
                script_name))
            os.system(batch_submit.format(
                queue, num_workers, j, j, setting_names[setting_i]+str(j),
                script_name))


if __name__ == '__main__':
    kwargs = parse_submit_args()
    print(kwargs)
    main(**kwargs)
