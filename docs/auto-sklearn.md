# [Auto-Sklearn](https://automl.github.io/auto-sklearn/)

## 1. Preprocessing

### 1.1 Encoding
---

By default if you provide a DataFrame with `dtype`s set or you provide the
`feat_type` argument to [AutoSklearnClassifier](https://automl.github.io/auto-sklearn/master/api.html#autosklearn.classification.AutoSklearnClassifier)
the library will use those to determine the appropriate encoding. 

It uses the  `LabelEncoder` for categorical values, the `OrdinalEncoder` for ordinal values,
leaves numerical values as-is. It also encodes boolean values using the `OrdinalEncoder`.<sup>[[1](https://github.com/automl/auto-sklearn/blob/master/autosklearn/data/validation.py#L75-L76)]</sup>

### 1.2 Available Preprocessors
---

The preprocessing steps done by `auto-sklearn` are data preprocessing and feature preprocessing. The data preprocessing step cannot be turned off as of yet.<sup>[[2](https://automl.github.io/auto-sklearn/master/manual.html#turning-off-preprocessing)]</sup> 

#### 1.2.1 Data Preprocessing

The data preprocessors used can be any of the following for binary classification tasks:

1. one-out-of-k encoding
2. imputation
3. balancing
4. rescaling


#### 1.2.2 Feature Preprocessing

Feature preprocessing can be any of the following for binary classification tasks<sup>[[3](https://papers.nips.cc/paper/2015/file/11d0e6287202fced83f79975ec59a3a6-Supplemental.zip)]</sup>:

1. no preprocessing
2. densifier
3. extremely random trees preprocessor
4. kernel PCA
5. random kitchen sinks
6. linear SVM preprocessor
7. nystroem sampler
8. random trees embedded
9. select percentile
10. select rates
11. truncated SVD

None of these are used in my implementation. This is achieved by calling `AutoSklearnClassifier` with the keyword argument `include_preprocessors=['no_preprocessing']`.


## 2. Models

### 2.1 Classifiers
---

The following classifiers are considered for binary classification tasks<sup>[[3](https://papers.nips.cc/paper/2015/file/11d0e6287202fced83f79975ec59a3a6-Supplemental.zip)]</sup>:

1. AdaBoost
2. Bernoulli naïve Bayes
3. decision tree
4. extremely random trees
5. Gaussian naïve Bayes
6. gradient boosting
7. k-Nearest-Neighbours (kNN)
8. LDA
9. linear SVM
10. kernel SVM
11. multinomial naïve Bayes
12. passive aggressive
13. QDA
14. random forest
15. SGD

A subset of these can be selected by passing the keyword argument `include_estimators=estimator_list` to the `AutoSklearnClassifier` class.

Currently, I place no limitations on the classifiers that are considered.

### 2.2 Other Tasks
---

In additional to binary classification, `auto-sklearn` supports multi-label classification and regression. I have yet to test either of these.

## 3. Using AutoSklearnClassifier

### 3.1 Pseudo Code
---

The pseudo-code for the implementation of the feature engineering aide is as follows:

```python
for experiment in experiments:
    for preprocessor_class in [None, SelectKBest, SelectPercentile, RFE]:
        classifier = AutoSklearn()  # Arguments described in `3.2`

        features_selected = experiment.features
        if preprocessor_class:
            preprocessor = preprocessor_class().fit(X, y)
            features_selected = preprocessor.get_features()

        classifier.fit(
            X[features_selected], y, train_X[features_selected], train_Y)

        leave_one_out = LeaveOneGroupOut()
        for train_idx, test_idx in leave_one_out.split():
            ... # partition data appropriately
            classifier.refit(x_train, y_train)
            y_hat = classifier.predict(x_test)

            get_metrics(y_test, y_hat)

        classifier.refit(X[features_selected], y)

        holdout_y_hat = classifier.predict(holdout_x[features_selected])
        get_metrics(holdout_y, holdout_y_hat)

        write_results_to_file()
```

Each experiment is instantiated by an experiment `TOML` file.

### 3.2 Arguments
---

The arguments used in the Feature Engineering Aide are:

```python
classifier = AutoSklearnClassifier(
    include_preprocessors=['no_preprocessing'],
    time_left_for_this_task=TASK_TIME,
    per_run_time_limit=TIME_PER_RUN,
    n_jobs=N_JOBS,
    # Each one of the N_JOBS jobs is allocated MEMORY_LIMIT
    memory_limit=MEMORY_LIMIT,
    resampling_strategy=LeaveOneGroupOut,
    resampling_strategy_arguments={'groups': experiment.groups},
    scoring_functions=metrics_as_scorers,
    seed=experiment.seed,
)
```

To see the full list of arguments with their descriptions see the [AutoSklearnClassifier docunentation](https://automl.github.io/auto-sklearn/master/api.html#autosklearn.classification.AutoSklearnClassifier).


### 3.3 Custom Feature Selection Step
---

After the `AutoSklearnClassifier` is instantiated as in `3.2` the currently considered preprocessor is instantiated and used to select the features that should be used for the current experiment step.

The preprocessors currently used are: `None`, `SelectKBest`, `SelectPercentile`, and `Recursive Feature Elimination` using a `DecisionTree`.

Each is used to get a list of feature labels to select data from the training data.

### 3.4 Results
---

Results are calculated after the classifier fitting process is finished. The created ensemble is used to get the `accuracy`, `f1 score`, and `Matthew's correlation coefficient` for each group using the `LeaveOneGroupOut` train-test splitter.

Finally, the full dataset is refit on the original data (including feature selection) and then tested against the holdout dataset.