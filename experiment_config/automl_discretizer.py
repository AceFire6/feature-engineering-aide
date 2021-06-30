from typing import Optional

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SIGNED_DATA, SPARSE, UNSIGNED_DATA
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformIntegerHyperparameter
from numpy.typing import ArrayLike
from sklearn import preprocessing


class KBinsDiscretizer(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_bins: int = 5, encode: str = 'onehot', strategy: str = 'quantile', random_state=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.random_state = random_state
        self.preprocessor = preprocessing.KBinsDiscretizer(n_bins, encode=encode, strategy=strategy)

    def fit(self, X: ArrayLike, y: ArrayLike) -> 'KBinsDiscretizer':
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[dict] = None) -> dict:
        return {
            'shortname': 'KBins',
            'name': 'KBinsDiscretizer',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
            'output': (INPUT, SPARSE, DENSE),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[dict] = None) -> ConfigurationSpace:
        n_bins = UniformIntegerHyperparameter('n_bins', lower=2, upper=15, default_value=5)
        encode = CategoricalHyperparameter(
            'encode',
            ['onehot', 'onehot-dense', 'ordinal'],
            default_value='onehot',
        )
        strategy = CategoricalHyperparameter(
            'strategy',
            ['uniform', 'quantile', 'kmeans'],
            default_value='quantile',
        )

        config_space = ConfigurationSpace()
        config_space.add_hyperparameters((
            n_bins,
            encode,
            strategy,
        ))
        return config_space
