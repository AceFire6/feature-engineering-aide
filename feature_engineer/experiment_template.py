from logging import Logger
import sys

from feature_engineer.experiment_config.experiment import Experiment
from feature_engineer.experiment_runner.runner import ExperimentRunner
from feature_engineer.experiment_runner.types import LabelledResults


class ExperimentName(ExperimentRunner):
    def experiment(self, experiment: Experiment, logger: Logger) -> LabelledResults:
        pass


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_results = ExperimentName(*experiment_input_paths, use_random_seeds=True)
    print(f'Experiments finished - {experiment_results}')
