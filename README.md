# Feature Engineering Aide

---

## Why?
This project was developed as part of my Master's thesis 
at the University of Cape Town.

The goal of the project is to compare the results of a project done 
by Nudelman et al. with those produced by Automated Machine Learning systems.

The purpose of this is to determine if AutoML is able to beat a human
"expert" at the task of feature engineering.

This is done using `auto-sklearn`. We establish experiments that conform
to those done by Nudelman et al. and run them using AutoML and then compare
the results to the results of the original experiments.

## How?

To run this, you can either install the requirements locally:
```shell
$ pip install -r requirements.txt
```

or if these don't work (on MacOS for example)

```shell
$ docker build -t feature-engineering-aide:latest .
```
