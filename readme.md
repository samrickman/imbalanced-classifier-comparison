# Imbalanced data machine learning classifiers plot

## What is this?

This is the Python code required to generate the following plot, which compares some machine learning classifiers:

![](https://raw.githubusercontent.com/samrickman/imbalanced-classifier-comparison/main/plots/algorithms.png)

This plot is based on the one created by in the [Sci-Kit Learn documentation](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html). That was created by Gaël Varoquaux and Andreas Müller, and modified for documentation by Jaques Grobler.

## Why create this?

I created this for a presentation as a way of graphically illustrating the differences between machine learning algorithms, without explaining the algorithms themselves.

In my real world example (which has data that cannot be published), I have extremely imbalanced data. I wanted to see how this compared to the comparisons in the documentation. 

1. The comparison in the Sci-Kit Learn documentation use balanced data (i.e. 0.5/0.5 split between binary classes). This plot uses unbalanced data (0.93/0.07) split.
2. The comparison in the documentation only contains Accuracy as a metric (which is reasonable for balanced data). This also has [F1 score](https://en.wikipedia.org/wiki/F-score), the harmonised mean of precision and recall.

There are also cosmetic differences, e.g. the plots in the documentation use [contourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf), while these use [pcolormesh](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html).

## To do:

This only currently compares six classifiers: logistic regression, random forest, bagging, gradient boosting, gaussian process and a sequential neural network (multi-layer perceptron). I plan to extend this in due course.

# How to run

Assuming you have Python, pip and virtualenv installed:

## How to run locally

### Installation 

To run, you will need to have [Python (3.9+)](https://www.python.org/downloads/), [pip](https://pip.pypa.io/en/stable/installing/), [virtualenv](https://pypi.org/project/virtualenv/) and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed. 

Once this is done, open a terminal to the folder you want to download the repo to and:

On Windows:
```
git clone https://github.com/samrickman/imbalanced-classifier-comparison
python -m venv ./venv
.\venv\Scripts\activate.ps1
pip install -r .\requirements.txt
python .\main.py
```

On Mac/Linux:
```
git clone https://github.com/samrickman/imbalanced-classifier-comparison
python -m venv ./venv
source venv/bin/activate
pip install -r .\requirements.txt
python .\main.py
```

# License

This code is based on other code with a 3 clause BSD license, and so has the same license. 