
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

import pandas as pd
from sklearn.metrics import f1_score

df_circles = pd.read_csv("./csv/df_circles.csv")
df_moons = pd.read_csv("./csv/df_moons.csv")
df = pd.read_csv("./csv/df_linear.csv")

datasets = [
    (df[['x', 'y']].values, df['class'].values),
    (df_moons[['x', 'y']].values, df_moons['class'].values),
    (df_circles[['x', 'y']].values, df_circles['class'].values)
]


names = [
    "Logistic regression",
    "Random Forest",
    "Bagging",
    "Gradient Boosting",
    "Gaussian Process",
    "Neural Network",
]

classifiers = [
    LogisticRegression(class_weight='balanced', verbose=True),
    BaggingClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GradientBoostingClassifier(),
    GaussianProcessClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
]


def draw_classification_plot():

    figure = plt.figure(figsize=(12, 9))
    figure.patch.set_facecolor('white')
    fontsize = 12
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data", fontsize=fontsize)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                   cmap=cm_bright, edgecolors="k", alpha=0.2)
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.2, edgecolors="k"
        )
        ax.text(
            x_max,
            y_min + 0.3,
            "F1:",
            size=15,
            horizontalalignment="left",
        )
        ax.text(
            x_max,
            y_min + 1.2,
            "Acc:",
            size=15,
            horizontalalignment="left",
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_pred=y_pred, y_true=y_test)

            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5, plot_method='pcolormesh'
            )

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:,
                                       1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name, fontsize=fontsize)
            ax.text(
                x_max,
                y_min + 0.3,
                ("%.2f" % f1).lstrip("0"),
                size=15,
                horizontalalignment="left",
            )
            ax.text(
                x_max,
                y_min + 1.2,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="left",
            )
            i += 1

    plt.tight_layout()
    if not os.path.exists("./plots/"):
        os.makedirs("./plots/")
    plt.savefig("./plots/algorithms.png")
    plt.show()


if __name__ == "__main__":
    draw_classification_plot()
