import sys

from experiment_config.experiment import Experiment, parse_experiment_paths
from experiment_config.settings import MEMORY_LIMIT, N_JOBS, TASK_TIME, TIME_PER_RUN, TOTAL_MEMORY_LIMIT


def run_experiments(experiments: list[Experiment]) -> None:
    total_experiments = len(experiments)
    print(
        f'Running {total_experiments} experiment(s)!',
        'Config:',
        f'\tn_jobs = {N_JOBS}',
        f'\ttotal_time = {TASK_TIME}',
        f'\ttime_per_run = {TIME_PER_RUN}',
        f'\ttotal_memory_limit = {TOTAL_MEMORY_LIMIT}',
        f'\tmemory_limit_per_run = {MEMORY_LIMIT}',
        sep='\n',
    )


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_list = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiment_list)
    print('Experiments finished!')
