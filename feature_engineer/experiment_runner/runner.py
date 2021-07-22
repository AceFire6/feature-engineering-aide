from datetime import datetime
import logging
from logging import FileHandler, Formatter, Logger
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Optional,
    Sequence,
    Union,
)

import joblib
import orjson

from feature_engineer.experiment_config.experiment import Experiment

from .runner_logging import setup_logger
from .settings import DATE_FORMAT
from .types import ExperimentInfo, ExperimentRun, LabelledResults
from .utils import hook_function


class ExperimentRunner:
    name: str = None
    experiments: list[Experiment]

    runner_logging_path: Optional[Path] = None
    console_log_level: ClassVar[int] = logging.INFO
    experiment_log_format: ClassVar[Optional[str]] = None
    log_to_file: ClassVar[bool] = True

    def __init__(
        self,
        *experiment_paths: str,
        runner_logging_path: Optional[Union[str, Path]] = None,
        run_experiments_n_times: int = 1,
        seeds_for_experiment_runs: Optional[Sequence[int]] = None,
        use_random_seeds: bool = False,
    ):
        self.run_start = datetime.now()
        self.name = self.name or self.__class__.__name__
        self.run_experiments_n_times = run_experiments_n_times
        self.use_random_seeds = use_random_seeds

        self.seeds_for_experiment_runs = seeds_for_experiment_runs
        if seeds_for_experiment_runs is not None:
            if use_random_seeds is True:
                raise ValueError('Cannot set seeds_for_experiment_runs and use_random_seeds at the same time!')

            if len(seeds_for_experiment_runs) != run_experiments_n_times:
                raise ValueError(
                    'seeds_for_experiment_runs len should be equal to run_experiment_n_times - '
                    f'{len(seeds_for_experiment_runs)=} != {run_experiments_n_times=}',
                )

        self._experiment_paths = experiment_paths
        self._additional_experiment_config = {'use_random_seed': use_random_seeds}
        self.experiments = Experiment.from_files(*experiment_paths, global_config=self._additional_experiment_config)

        # This will be set on the first access of self.logger
        self._logger = None

        if isinstance(runner_logging_path, str):
            runner_logging_path = Path(runner_logging_path)

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

        return self._logger

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

    def write_experiment_data(
        self,
        experiment: Experiment,
        experiment_data: Union[LabelledResults, ExperimentRun, ExperimentInfo],
        label: Optional[str] = None,
    ) -> None:
        self.logger.info(f'Writing results for {experiment.name} to file - {experiment_data}')
        experiment_results_file_path = self.get_experiment_results_file(experiment)

        with experiment_results_file_path.open('ab') as results_file:
            if label is not None:
                results_file.write(f'{label}: '.encode())

            results_file.write(
                orjson.dumps(
                    experiment_data,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
                ),
            )

    def write_model(self, experiment: Experiment, model: Any, name: str = '') -> None:
        name = f'_{name}' if name else name

        results_path = self.get_results_path(experiment)
        model_filename = f'{experiment.name}_{experiment.start_time}_model{name}.joblib.lzma'

        model_dump_file = results_path / model_filename

        joblib.dump(model, filename=model_dump_file)

    def write_cv_results(self, experiment: Experiment, cv_results: dict) -> None:
        results_path = self.get_results_path(experiment)

        model_filename = f'{experiment.name}_{experiment.start_time}_cv_results.joblib.lzma'
        model_dump_file = results_path / model_filename

        joblib.dump(cv_results, filename=model_dump_file)

    def run_experiment(self, experiment: Experiment) -> ExperimentInfo:
        run_results = []

        for run_index in range(self.run_experiments_n_times):
            run_start_datetime = datetime.now()
            run_start = f'{run_start_datetime:%Y-%m-%d_%H:%M:%S}'
            experiment.start_time = run_start

            run_number = run_index + 1
            run_counter = f'{run_number} / {self.run_experiments_n_times}'
            self.logger.info(f'Running {experiment=} - run number {run_counter} - {run_start}')

            # Set the seed for the current run
            seed_to_set = None
            if self.seeds_for_experiment_runs:
                seed_to_set = self.seeds_for_experiment_runs[run_index]

            experiment_run_seed = experiment.reset_seed(seed_to_set)
            logger = self.get_experiment_logger(experiment)

            experiment_result = self.experiment(experiment, logger)

            run_end_datetime = datetime.now()
            run_end = f'{run_end_datetime:%Y-%m-%d_%H:%M:%S}'
            self.logger.info(
                f'Experiment {experiment.name} [{run_counter}] finished! - {run_end} - Results: {experiment_result}',
            )

            run_result: ExperimentRun = {
                'run_number': run_number,
                'seed': experiment_run_seed,
                'time_taken': (run_end_datetime - run_start_datetime).total_seconds(),
                'results': experiment_result,
            }
            run_results.append(run_result)

            self.write_experiment_data(experiment, run_result, label=f'Run {run_number}')

        return {
            'experiment': experiment,
            'runs': run_results,
        }

    def run_experiments(self) -> list[ExperimentInfo]:
        experiment_names = ', '.join(experiment.name for experiment in self.experiments)
        self.logger.info(f'Starting experiments: {experiment_names}')

        experiment_results = []
        for experiment in self.experiments:
            experiment_result = self.run_experiment(experiment)
            experiment_results.append(experiment_result)

        return experiment_results

    # Runner hooks - override these to add additional functionality
    def experiment(self, experiment: Experiment, logger: Logger) -> LabelledResults:
        raise NotImplementedError(
            'Missing experiment function - this should be overridden to define how the experiment is run',
        )

    def _before_run_all_experiments(self) -> None:
        self.logger.debug('Running _before_run_all_experiments')

    def _after_run_all_experiments(self) -> None:
        self.logger.debug('Running _after_run_all_experiments')

    def _before_run_experiment(self, experiment: Experiment, **kwargs) -> None:
        self.logger.debug(f'Running _before_run_experiment - {experiment.name}')

    def _after_run_experiment(self, experiment: Experiment, **kwargs) -> None:
        self.logger.debug(f'Running _after_run_experiment - {experiment.name}')
