from functools import partial

from environs import Env
from prompt_toolkit.styles import style_from_dict
from pygments.token import Token
from PyInquirer import prompt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier


try:
    from sklearn.naive_bayes import ComplementNB as NaiveBayes
except ImportError:
    from sklearn.naive_bayes import GaussianNB as NaiveBayes
    print("Can't use ComplementNB - using GaussianNB")


env = Env()

prompt_style = style_from_dict({
    Token.Separator: '#6C6C6C',
    Token.QuestionMark: '#FF9D00 bold',
    Token.Selected: '#5F819D',
    Token.Pointer: '#FF9D00 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#5F819D bold',
    Token.Question: '',
})

styled_prompt = partial(prompt, style=prompt_style)

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

DATA_TYPE_CHOICES = {
    'Categorical': 'category',
    'Boolean': 'bool',
}

N_JOBS = env.int('FEA_N_JOBS', default=4)
TOTAL_MEMORY_LIMIT = env.int('FEA_TOTAL_MEMORY_LIMIT', default=None)
TASK_TIME = env.int('FEA_TASK_TIME', default=180)
TIME_PER_RUN = TASK_TIME // 10

# Default in auto-sklearn AutoSklearnClassifier
MEMORY_LIMIT = 3072
if TOTAL_MEMORY_LIMIT is not None:
    MEMORY_LIMIT = TOTAL_MEMORY_LIMIT // N_JOBS
