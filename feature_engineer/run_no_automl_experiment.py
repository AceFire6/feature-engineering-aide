from logging import Logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier

from feature_engineer.experiment_cli import run_experiments
from feature_engineer.experiment_config.experiment import Experiment
from feature_engineer.experiment_runner.runner import ExperimentRunner
from feature_engineer.experiment_runner.types import ExperimentResult
from feature_engineer.experiment_runner.utils import get_metric_results_five_number_summary


class AtRiskNoAutoML(ExperimentRunner):
    def experiment(self, experiment: Experiment, logger: Logger) -> dict[str, ExperimentResult]:
        labelled_results = {}

        features_selected = experiment.prediction_data_columns
        x_train = experiment.X[features_selected].copy()
        y_train = experiment.y.copy()

        categorical_nb = CategoricalNB()
        random_forest = RandomForestClassifier(random_state=experiment.seed)
        decision_tree = DecisionTreeClassifier(random_state=experiment.seed)

        leave_one_out = LeaveOneGroupOut()

        for classifier in [categorical_nb, random_forest, decision_tree]:
            test_train_splitter = leave_one_out.split(x_train, y_train, experiment.groups)

            for train_index, test_index in test_train_splitter:
                x_train_split, x_test_split = experiment.get_split(x_train, train_index, test_index)
                y_train_split, y_test_split = experiment.get_split(y_train, train_index, test_index)

                # Value of the group used for splitting
                split_value = experiment.groups.iloc[test_index].unique()[0]

                classifier.fit(x_train_split, y_train_split)
                y_hat = classifier.predict(x_test_split)

                for metric, metric_function in experiment.metrics.items():
                    metric_result = metric_function(y_test_split, y_hat)
                    experiment.add_result(metric, metric_result, label=split_value)

            classifier.fit(x_train, y_train)
            holdout_prediction = classifier.predict(experiment.holdout_x[features_selected])
            holdout_mcc_result = matthews_corrcoef(experiment.holdout_y, holdout_prediction)
            holdout_f1_result = f1_score(experiment.holdout_y, holdout_prediction)
            holdout_balanced_accuracy_result = accuracy_score(experiment.holdout_y, holdout_prediction)
            classification_text_report = classification_report(experiment.holdout_y, holdout_prediction)

            experiment_results: ExperimentResult = {
                'features_used': list(features_selected),
                'train_test_metric_results': experiment.metric_results,
                'train_test_metric_results_summary': get_metric_results_five_number_summary(
                    experiment.metric_results,
                ),
                'holdout_metric_results': {
                    'mcc': holdout_mcc_result,
                    'f1': holdout_f1_result,
                    'balance_accuracy': holdout_balanced_accuracy_result,
                },
                'classification_report': classification_text_report,
                'extra_data': {},
            }
            labelled_results[classifier.__class__.__name__] = experiment_results

            self.write_model(experiment, classifier, name=classifier.__class__.__name__)

        return labelled_results


if __name__ == '__main__':
    run_experiments(AtRiskNoAutoML)
