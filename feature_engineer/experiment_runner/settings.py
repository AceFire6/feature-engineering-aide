from functools import partial
import logging

from environs import Env
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier


try:
    from sklearn.naive_bayes import ComplementNB as NaiveBayes
except ImportError:
    from sklearn.naive_bayes import GaussianNB as NaiveBayes
    print("Can't use ComplementNB - using GaussianNB")


env = Env()

with env.prefixed('FEA_'):
    LOG_LEVEL = env.log_level('LOG_LEVEL', default=logging.DEBUG)
    LOG_TO_FILE = env.bool('LOG_TO_FILE', default=True)

    N_JOBS = env.int('N_JOBS', default=4)
    TOTAL_MEMORY_LIMIT = env.int('TOTAL_MEMORY_LIMIT', default=None)
    TASK_TIME = env.int('TASK_TIME', default=180)
    TIME_PER_RUN = TASK_TIME // 10

# Default in auto-sklearn AutoSklearnClassifier
MEMORY_LIMIT = 3072
if TOTAL_MEMORY_LIMIT is not None:
    MEMORY_LIMIT = TOTAL_MEMORY_LIMIT // N_JOBS


DATE_FORMAT = '%Y_%m_%dT%H_%M_%S'
LOG_FORMAT = '{name} - {created} - {levelname} - [{funcName}] {message} - {pathname}:{lineno}'
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, style='{')


SUPPORTED_CLASSIFIERS = {
    'Naive Bayes': NaiveBayes,
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
}


SUPPORTED_METRICS = {
    'Accuracy': accuracy_score,
    'F1 Score': f1_score,
    "Matthew's Correlation Coefficient": matthews_corrcoef,
}

SUPPORTED_FEATURE_PREPROCESSORS = {
    'no_preprocessor': None,
    'SelectKBest': SelectKBest,
    'SelectPercentile': SelectPercentile,
    'DecisionTreeRFE': partial(RFE, estimator=DecisionTreeClassifier()),
}

ORDINAL = 'ordinal'
DATA_TYPE_CHOICES = {
    'Boolean': 'bool',
    'Ordinal': ORDINAL,
    'Categorical': 'categorical',
}
