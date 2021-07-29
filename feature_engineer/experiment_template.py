from logging import Logger

from feature_engineer.experiment_cli import run_experiments
from feature_engineer.experiment_config.experiment import Experiment
from feature_engineer.experiment_runner.runner import ExperimentRunner
from feature_engineer.experiment_runner.types import ExperimentResult


class ExperimentName(ExperimentRunner):
    def experiment(self, experiment: Experiment, logger: Logger) -> dict[str, ExperimentResult]:
        pass


if __name__ == '__main__':
    run_experiments(ExperimentName)
