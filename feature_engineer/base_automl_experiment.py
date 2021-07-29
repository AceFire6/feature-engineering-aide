from datetime import datetime
from logging import Logger

from autosklearn.estimators import AutoSklearnClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut

from feature_engineer.experiment_config.experiment import Experiment
from feature_engineer.experiment_runner import settings
from feature_engineer.experiment_runner.runner import ExperimentRunner
from feature_engineer.experiment_runner.types import ExperimentResult
from feature_engineer.experiment_runner.utils import get_metric_results_five_number_summary


class BaseAutoMLExperiment(ExperimentRunner):
    def get_classifier(self, experiment: Experiment) -> AutoSklearnClassifier:
        raise NotImplementedError

    def experiment(self, experiment: Experiment, logger: Logger) -> dict[str, ExperimentResult]:
        labelled_results = {}

        preprocessor_count = len(experiment.feature_preprocessors)
        for p_index, (preprocessor_name, preprocessor_class) in enumerate(experiment.feature_preprocessors.items()):
            preprocessor_start = datetime.now()
            preprocessor_counter = f'[{p_index + 1}/{preprocessor_count}]'
            logger.info(
                f'Running preprocessor {preprocessor_name} '
                f'{preprocessor_counter} - experiment {experiment.name} - {preprocessor_start:%Y-%m-%d_%H:%M:%S}',
            )

            classifier = self.get_classifier(experiment)

            features_selected = experiment.prediction_data_columns
            if preprocessor_class is not None:
                preprocessor = preprocessor_class().fit(experiment.X, experiment.y)
                features_selected_mask = preprocessor.get_support()
                features_selected = list(experiment.X.columns[features_selected_mask])

            training_data = experiment.X[features_selected].copy()
            targets = experiment.y.copy()
            classifier.fit(
                X=training_data,
                y=targets,
                X_test=experiment.holdout_x[features_selected],
                y_test=experiment.holdout_y,
                dataset_name=f'{experiment.name} - {preprocessor_name}',
            )

            leave_one_out = LeaveOneGroupOut()
            test_train_splitter = leave_one_out.split(training_data, targets, experiment.groups)
            for train_index, test_index in test_train_splitter:
                x_train, x_test = experiment.get_split(training_data, train_index, test_index)
                y_train, y_test = experiment.get_split(targets, train_index, test_index)

                # Value of the group used for splitting
                split_value = experiment.groups.iloc[test_index].unique()[0]

                classifier.refit(x_train, y_train)
                y_hat = classifier.predict(x_test)

                for metric, metric_function in experiment.metrics.items():
                    metric_result = metric_function(y_test, y_hat)
                    experiment.add_result(metric, metric_result, label=split_value)

            classifier.refit(training_data, targets)
            holdout_prediction = classifier.predict(experiment.holdout_x[features_selected])
            holdout_mcc_result = matthews_corrcoef(experiment.holdout_y, holdout_prediction)
            holdout_f1_result = f1_score(experiment.holdout_y, holdout_prediction)
            holdout_balanced_accuracy_result = accuracy_score(experiment.holdout_y, holdout_prediction)
            classification_text_report = classification_report(experiment.holdout_y, holdout_prediction)

            experiment_results: ExperimentResult = {
                'features_used': list(features_selected),
                'train_test_metric_results': experiment.metric_results,
                'train_test_metric_results_summary': get_metric_results_five_number_summary(experiment.metric_results),
                'holdout_metric_results': {
                    'mcc': holdout_mcc_result,
                    'f1': holdout_f1_result,
                    'balance_accuracy': holdout_balanced_accuracy_result,
                },
                'classification_report': classification_text_report,
                'extra_data': {
                    'statistics': classifier.sprint_statistics(),
                },
            }
            labelled_results[preprocessor_name] = experiment_results

            self.write_cv_results(experiment, classifier.cv_results_, name=preprocessor_name)
            self.write_model(
                experiment,
                classifier.get_models_with_weights(),
                name=f'{preprocessor_name}_final_ensemble',
            )

        return labelled_results

    def _before_run_experiment(self, experiment: Experiment, **kwargs) -> None:
        run_start_datetime = datetime.now()
        run_start = f'{run_start_datetime:%Y-%m-%d_%H:%M:%S}'
        experiment.start_time = run_start

        experiment_logger = self.get_experiment_logger(experiment)
        config_log = '\n'.join(
            [
                f'Running {len(self.experiments)} experiment(s)!',
                'Config:',
                f'\tn = {experiment.training_set_sample_size()}',
                f'\tn_jobs = {settings.N_JOBS}',
                f'\ttotal_time = {settings.TASK_TIME}',
                f'\ttime_per_run = {settings.TIME_PER_RUN}',
                f'\ttotal_memory_limit = {settings.TOTAL_MEMORY_LIMIT}',
                f'\tmemory_limit_per_run = {settings.MEMORY_LIMIT}\n',
            ]
        )
        experiment_logger.info(config_log)
