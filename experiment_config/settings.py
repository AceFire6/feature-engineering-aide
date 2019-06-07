from functools import partial

from prompt_toolkit.styles import style_from_dict
from pygments.token import Token
from PyInquirer import prompt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier


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
    'Naive Bayes': ComplementNB,
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
