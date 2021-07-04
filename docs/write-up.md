# Overview

## Input Components

### Experiment Files

Each experiment I have currently is configured as a TOML file. Each file can be written by hand or generated using the tooling I wrote (`make_experiment.py`) in 2019. It is not currently updated with the new data schema. But writing it by hand is simple.

As an example here is the `at-risk.toml` with annotations in the form of comments.

```toml
# Added when the file is generated
timestamp = "2019-06-10T02:10:17.563175"
# The seed used by numpy & auto-sklearn
# If we keep it set then the runs should be deterministic
random_seed = 0
# The name of the experiment
experiment = "at_risk"
# The data file to use for the experiment
data_source = "./data_files/GradFull.csv"
# The classifiers we are willing to consider/use
classifiers = [ "Naive Bayes", "Decision Tree", "Random Forest",]
# The metrics we want to be reported
metrics = [ "Accuracy", "F1 Score", "Matthew's Correlation Coefficient",]

[data]
# The features that should be used from the dataset during classification
selected_features = [ "EngBin", "AveSciBin", "AdMathAttempt",]
# The target feature
target = "GradBin"
# The feature to perform LeaveOneGroupOut on
data_split_column = "Application Year"
# The holdout set value
holdout_split_condition = 2016
# Values to be considered True in the dataset
true_values = [ "Yes", "Pass",]
# Values to be considered False in the dataset
false_values = [ "No", "Fail",]

[data.features]
# Features that should be treated as boolean
bool = ["GradBin", "AdMathAttempt",]

# Features that should be treated as Ordinal
[data.features.ordinal]
# The ordinal values in order for the given feature
EngBin = ["Low", "Mid", "High",]
AveSciBin = ["Low", "Mid", "High",]

# Values to be considered NA per feature
[data.na_values]
EngBin = [ "*",]
AveSciBin = [ "*",]
GradBin = [ "*",]
```

This file is passed in as an input to my experiment runner. The experimental data is read in and configured according to the settings set in the `TOML` file. Multiple experiments can be run sequentially by passing in more experiment files.

### Data Processing

The experiment files are all parsed to generate `Experiment` objects that store and deal with all the data and configuration for each experiment file. The initialisation of each `Experiment` object involves reading in the dataset using `Pandas.read_csv` with all the columns (except boolean) `dtypes` set appropriately. Then any rows with `NA` values for one of features in the `selected_features` list is dropped.

Once the dataset is read in the `boolean` values have their `dtype` set (they can't handle `NA` values). 

Following that if there are any `ordinal` features are encoded by converting them to `integers` based on the provided ordering. eg. 

```python
eng_bin = ["Low", "Mid", "High",]
eng_bin_ordinal = [1, 2, 3]
```

This retains the "information" that `Low < Mid < High`.

Then if there are any `categorical` features they're encoded using `scikit-learn`'s [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). This transforms each categorical column into a set of columns that represents each unique value as a separate column with either `0` or `1` depending on the original value. eg.

```python
column = ['A', 'B', 'B', 'C']

# Do OneHotEncoding

encoded_columns = [
    [1, 0, 0, 0], # column_A
    [0, 1, 1, 0], # column_B
    [0, 0, 0, 1], # column_C
]
```

These new encoded columns replace the original column in the `dataframe`. Each of these new columns is added to the `selected_features` list after removing the original feature. eg.

```python
selected_features == ['column']

# Do OneHotEncoding

selected_features == ['column_A', 'column_B', 'column_C']
```

Finally, the data is processed and the holdout set is removed from the available features and target dataset splits.

## Auto-sklearn Usage

Upon starting the experiment running with experiment files they're parsed as explained in *Input* above. Each experiment is run in sequence.

For each experiment each of the four (currently) preprocessor setups is used. No preprocessor, `SelectKBest`, `SelectPercentile`, and `Recursive Feature Elimination` using a `DecisionTree` as the underlying classifier.

For each of these preprocessors I create an `AutoSklearnClassifier`.

```python
classifier = AutoSklearnClassifier(
    # prevent it doing preprocessing
    include_preprocessors=['no_preprocessing'],
    # The total time left for this classifier to find the best ensemble
    time_left_for_this_task=TASK_TIME,
    # The time each model has to train
    per_run_time_limit=TIME_PER_RUN,
    # The number of tasks to run for fitting and ensembling
    n_jobs=N_JOBS,
    # Each one of the N_JOBS jobs is allocated MEMORY_LIMIT
    memory_limit=MEMORY_LIMIT,
    # Which strategy the classifier uses for resampling
    resampling_strategy=LeaveOneGroupOut,
    resampling_strategy_arguments={'groups': experiment.groups},
    # Custom metric (currently not used)
    metric=metric_scorer,
    # The functions used to generate scores for display/writing to file
    scoring_functions=metrics_as_scorers,
    seed=experiment.seed,
)
```

Then the preprocessor strategy is used to determine which features to select from the training data to fit the classifier.

After fitting the classifier is used to generate the metric results using the `LeaveOneGroupOut` train/test splitter.

Finally, these resuts are written to a file and the iteration continues until all the experiments have been run for preprocessors.

## Output

The output is currently the `F1 Score`, `Accuracy`, and `Matthew's Correlation Coefficient` scores per preprocessor per experiment in a text file. Each text file also contains all the configuration used for that experiment run.

I have yet to add validation using the holdout set.
