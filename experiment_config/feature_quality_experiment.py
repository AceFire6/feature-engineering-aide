import sys

from experiment_config.experiment import Experiment, parse_experiment_paths


def run_experiments(experiments: list[Experiment]) -> None:
    pass


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_list = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiment_list)
    print('Experiments finished!')
