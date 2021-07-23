from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneGroupOut

from feature_engineer.base_automl_experiment import BaseAutoMLExperiment
from feature_engineer.experiment_cli import run_experiments
from feature_engineer.experiment_config.experiment import Experiment
from feature_engineer.experiment_runner import settings


class AutoMLFeatureQualityExperiment(BaseAutoMLExperiment):
    def get_classifier(self, experiment: Experiment) -> AutoSklearnClassifier:
        metrics_as_scorers = [make_scorer(name, score_func) for name, score_func in experiment.metrics.items()]

        return AutoSklearnClassifier(
            include_estimators=['decision_tree', 'random_forest', 'gaussian_nb'],
            time_left_for_this_task=settings.TASK_TIME,
            per_run_time_limit=settings.TIME_PER_RUN,
            n_jobs=settings.N_JOBS,
            # Each one of the N_JOBS jobs is allocated MEMORY_LIMIT
            memory_limit=settings.MEMORY_LIMIT,
            resampling_strategy=LeaveOneGroupOut,
            resampling_strategy_arguments={'groups': experiment.groups},
            ensemble_size=1,
            # metric=metric_scorer,
            scoring_functions=metrics_as_scorers,
            seed=experiment.seed,
        )


if __name__ == '__main__':
    run_experiments(AutoMLFeatureQualityExperiment)
