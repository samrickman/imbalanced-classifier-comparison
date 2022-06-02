from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def create_linearly_separable(num_samples, imbalance_ratio):
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_repeated=0,
        random_state=1,
        n_clusters_per_class=1,
        weights=[imbalance_ratio, 1-imbalance_ratio],
        n_samples=num_samples,
        flip_y=0,
        n_classes=2,
        class_sep=3
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    xy = np.c_[X, y]
    xy.shape

    df = pd.DataFrame(xy, columns=['x', 'y', 'class'])
    print("Linearly separable created")

    actual_num_in_majority, actual_num_in_minority = df['class'].value_counts(
    ).values

    print(
        f"Created linearly separable! Num in majority class: {actual_num_in_majority}. Num in minority class: {actual_num_in_minority}")

    return df


def create_not_linearly_separable(fn, num_in_majority_group, num_in_minority_group, noise):

    try:
        X, y = fn(
            random_state=1, n_samples=[num_in_majority_group, num_in_minority_group], noise=noise
        )
    except TypeError:  # make_blobs doesn't have a noise parameter - they're already noisy
        X, y = fn(
            random_state=1, n_samples=[num_in_majority_group, num_in_minority_group]
        )

    xy = np.c_[X, y]

    df = pd.DataFrame(xy, columns=['x', 'y', 'class'])

    df = df.sort_values('class')
    actual_num_in_majority, actual_num_in_minority = df['class'].value_counts(
    ).values
    shape = fn.__name__.replace("make_", "")
    print(f"Created {shape}! Num in majority class: {actual_num_in_majority}. Num in minority class: {actual_num_in_minority}")
    return {shape: df}

# Actually make the data


def create_all_data(
        not_linearly_separable_fns,
        num_samples,
        imbalance_ratio,
        num_in_majority_group,
        num_in_minority_group,
        noise):
    df_dict = {}
    df_dict['linear'] = create_linearly_separable(
        num_samples, imbalance_ratio)
    for fn in not_linearly_separable_fns:
        df_dict.update(create_not_linearly_separable(
            fn, num_in_majority_group, num_in_minority_group, noise))
    return df_dict


def draw_classification_plot(df_dict, classifier_names, classifiers, outfile, figsize=(12, 9)):

    datasets = [(df[['x', 'y']].values, df['class'].values)
                for df_name, df in df_dict.items()]

    figure = plt.figure(figsize=figsize)
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
        for name, clf in zip(classifier_names, classifiers):
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
    plt.savefig(f"./plots/{outfile}.png")
    plt.show()
