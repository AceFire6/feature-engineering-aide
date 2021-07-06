from datetime import datetime, timedelta
import logging
from logging import FileHandler, Formatter, Logger
from pathlib import Path
from typing import Any, ClassVar, Optional, TypedDict

import orjson

from feature_engineer.experiment_config.experiment import Experiment, parse_experiment_paths

from .runner_logging import setup_logger
from .settings import DATE_FORMAT
from .utils import hook_function


class ExperimentResult(TypedDict):
    experiment: Experiment
    time_taken: timedelta

    best_model: Any
    features_used: list[str]

    train_test_metric_results: dict[str, float]
    holdout_metric_results: dict[str, float]
    classification_report: str

    extra_data: dict[str, Any]


class ExperimentRunner:
    name: str = None
    experiments: list[Experiment]

    runner_logging_path: Optional[Path] = None
    console_log_level: ClassVar[int] = logging.INFO
    experiment_log_format: ClassVar[Optional[str]] = None
    log_to_file: ClassVar[bool] = True

    def __init__(self, *experiment_paths: str, runner_logging_path: Optional[Path] = None):
        self.run_start = datetime.now()
        self.name = self.name or self.__name__

        self._experiment_paths = experiment_paths
        self.experiments = parse_experiment_paths(experiment_paths)

        # This will be set on the first access of self.logger
        self._logger = None
        self.runner_logging_path = runner_logging_path

        self._experiment_loggers: dict[Experiment, Logger] = {}

    def __init_subclass__(cls, **kwargs):
        # Set up before all experiments hooks
        all_experiments_hook_decorator = hook_function(
            pre_hook=cls._before_run_all_experiments,
            post_hook=cls._after_run_all_experiments,
        )
        cls.run_experiments = all_experiments_hook_decorator(cls.run_experiments)

        # Set up before single experiment hooks
        experiment_hook = hook_function(pre_hook=cls._before_run_experiment, post_hook=cls._after_run_experiment)
        cls.run_experiment = experiment_hook(cls.run_experiment)

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            cwd_root = Path.cwd()
            default_logging_path = cwd_root / f'runner_log_{self.run_start:{DATE_FORMAT}}.log'
            logging_path = self.runner_logging_path or default_logging_path

            runner_file_handler = FileHandler(logging_path)
            self._logger = setup_logger(
                self.name,
                console_log_level=self.console_log_level,
                file_handler=runner_file_handler,
            )

        return self.logger

    def setup_experiment_logger(self, experiment: Experiment) -> Logger:
        default_log_format = (
            f'{{name}} - {{created}} - {{levelname}} - {experiment.name} - '
            '[{funcName}] {message} - {pathname}:{lineno}'
        )
        log_format = self.experiment_log_format or default_log_format
        log_formatter = Formatter(log_format, style='{')

        logger_kwargs = {'log_formatter': log_formatter}
        if self.log_to_file:
            results_path = self.get_results_path(experiment)
            file_handler = FileHandler(results_path / f'run_log_{experiment.start_time}.log')
            logger_kwargs['file_handler'] = file_handler

        logger = setup_logger(experiment.name, **logger_kwargs)

        return logger

    def get_experiment_logger(self, experiment: Experiment) -> Logger:
        if experiment not in self._experiment_loggers:
            experiment_logger = self.setup_experiment_logger(experiment)
            self._experiment_loggers[experiment] = experiment_logger

        return self._experiment_loggers[experiment]

    def get_results_path(self, experiment: Experiment = None) -> Path:
        # Make results directory if it doesn't exist
        experiment_results_folder = experiment.file_path.parent / 'results'
        run_start_str = self.run_start.strftime(DATE_FORMAT)

        # Relative to the experiment toml file
        # eg. ./experiments/results/ExperimentRunner/2021_07_04T19_10_26/at_risk/
        results_file_path = experiment_results_folder / self.name / run_start_str / experiment.name
        results_file_path.mkdir(parents=True, exist_ok=True)

        return results_file_path

    def get_experiment_results_file(self, experiment: Experiment) -> Path:
        results_path = self.get_results_path(experiment)
        results_file = f'{experiment.name}_results_{experiment.start_time}.txt'

        return results_path / results_file

    def write_experiment_results(self, experiment: Experiment, experiment_result: ExperimentResult) -> None:
        self.logger.info(f'Writing results for {experiment.name} to file - {experiment_result}')
        experiment_results_file_path = self.get_experiment_results_file(experiment)

        with experiment_results_file_path.open('wb') as results_file:
            results_file.write(orjson.dumps(experiment_result))

    def run_experiment(self, experiment: Experiment, logger: Optional[Logger] = None) -> ExperimentResult:
        raise NotImplementedError(
            'Missing run_experiment - this should be overridden to define how the experiment is run',
        )

    def run_experiments(self) -> dict[Experiment, ExperimentResult]:
        experiment_names = ', '.join(experiment.name for experiment in self.experiments)
        self.logger.info(f'Starting experiments: {experiment_names}')

        experiment_results = {}
        for experiment in self.experiments:
            self.logger.debug(f'Running {experiment=}')
            experiment_logger = self.get_experiment_logger(experiment)
            experiment_result = self.run_experiment(experiment, logger=experiment_logger)
            experiment_results[experiment] = experiment_result
            self.logger.info(f'Experiment {experiment.name} finished! - Results: {experiment_result}')
            self.write_experiment_results(experiment, experiment_result)

        return experiment_results

    # Runner hooks - override these to add additional functionality
    def _before_run_all_experiments(self):
        self.logger.debug('Running _before_run_all_experiments')

    def _after_run_all_experiments(self):
        self.logger.debug('Running _after_run_all_experiments')

    def _before_run_experiment(self):
        self.logger.debug('Running _before_run_experiment')

    def _after_run_experiment(self):
        self.logger.debug('Running _after_run_experiment')
