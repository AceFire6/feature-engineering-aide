from pathlib import Path
import typing
from typing import Type

import click

if typing.TYPE_CHECKING:
    from feature_engineer.experiment_runner.runner import ExperimentRunner


def run_experiments(experiment_class: Type['ExperimentRunner']) -> None:

    @click.command()
    @click.option('-n', '--run-experiments-n-times', type=click.INT, default=1)
    @click.option('--use-random-seeds', is_flag=True, type=click.BOOL, default=False)
    @click.argument('experiment_files', nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
    def _run_experiments(
        experiment_files: tuple[Path],
        use_random_seeds: bool,
        run_experiments_n_times: int,
    ) -> None:
        auto_ml_preprocessor_experiment_runner = experiment_class(
            *experiment_files,
            use_random_seeds=use_random_seeds,
            run_experiments_n_times=run_experiments_n_times,
        )

        experiment_results = auto_ml_preprocessor_experiment_runner.run_experiments()
        click.echo(f'Experiments finished - {experiment_results}')

    return _run_experiments()
