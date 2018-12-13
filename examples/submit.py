import pandas as pd
import os


# convert dictionay to cmd args in the form "--key value"
def dict_to_cmd_args(d):
    s = []
    for key, value in d.items():
        s.append("--"+key+" "+str(value))
    return " ".join(s)


def read_job_id_from_job_name(job_name):
    # find slurm job id based on given job name and read it
    job_id = os.popen('squeue --noheader --format %i --name {}'.format(job_name)).read()
    # since they are all array jobs, only take the job id not the array id
    dependency_job_id = job_id.split("_")[0]
    return dependency_job_id


def main():
    # create jobs file text in this script from which a temporary file will be created
    job_file = \
        "#!/bin/bash\n" \
        "# redirect the output/error to some files\n" \
        "#SBATCH -o /home/gemeinl/out/%A-%a.o\n" \
        "#SBATCH -e /home/gemeinl/error/%A-%a.e\n" \
        "export PYTHONPATH={}\n" \
        "source {}\n" \
        "python {} {}\n"

    configs_file = "/home/gemeinl/code/braindecodelazy/examples/configs.csv"
    # load all the configs to be run
    configs_df = pd.DataFrame.from_csv(configs_file)

    # specify python path, virtual env and python cript to be run
    python_path = '/home/gemeinl/code/braindecodelazy'
    virtual_env = 'anaconda3/bin/activate && conda activate braindecode3'
    python_file = '/home/gemeinl/code/braindecodelazy/examples/tuh_auto_diag_lazy.py'

    # specify queue, temporary job file and command to submit
    queue = "meta_gpu-ti"
    script_name = "/home/gemeinl/jobs/slurm/run_tmp.sh"
    batch_submit = "sbatch -p {} --array={}-{} --job-name=b_{}_j_{} {} {}"
    dependency = "--dependency=afterok:{}"

    n_parallel = 5  # number of jobs to run in parallel in a batch
    batch_i = 0  # batch_i current batch running
    i = 0  # i total number of run jobs

    # loop through all the configs
    for setting in configs_df:
        config = configs_df[setting].to_dict()
        # TODO: what if n_folds not in config?
        n = int(config["n_folds"])
        # create a tmp job file / job for every repetition / fold
        for j in range(n):
            # if this is not the very first job, increase batch_i whenever n_parallel jobs were submitted
            if i != 0 and i % n_parallel == 0:
                batch_i += 1

            config["seed"] = j
            cmd_args = dict_to_cmd_args(config)
            curr_job_file = job_file.format(python_path, virtual_env, python_file, cmd_args)

            # write tmp job file and submit it to slurm
            with open(script_name, "w") as f:
                f.writelines(curr_job_file)

            # when this is not the first batch, add dependecy on the previous batch
            dependency_job_name = "b_" + str(batch_i - 1) + "_j_" + str(i % n_parallel)
            dependency_job_id = read_job_id_from_job_name(dependency_job_name)
            dependency_term = "" if batch_i == 0 else dependency.format(dependency_job_id)
            print(batch_submit.format(queue, j, j, batch_i, i % n_parallel, dependency_term, script_name))
            os.system(batch_submit.format(queue, j, j, batch_i, i % n_parallel, dependency_term, script_name))
            i += 1


if __name__ == '__main__':
    # TODO: add arg parse
    main()
