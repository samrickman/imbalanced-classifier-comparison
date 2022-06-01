import numpy as np
import os
from sklearn.datasets import make_moons, make_circles, make_classification

import pandas as pd

NUM_SAMPLES = 10_000
IMBALANCE_RATIO = 0.93  # this will create ~9300 in one class and 700 in the other


def create_linearly_separable(num_samples=NUM_SAMPLES, imbalance_ratio=IMBALANCE_RATIO):
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
        weights=[imbalance_ratio, 1-imbalance_ratio],
        n_samples=num_samples
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    xy = np.c_[X, y]
    xy.shape

    df = pd.DataFrame(xy, columns=['x', 'y', 'class'])
    print("Linearly separable created")
    print("Num in categories:")
    print(df['class'].value_counts())
    return df


def create_moons(num_samples=NUM_SAMPLES, imbalance_ratio=IMBALANCE_RATIO):

    # Create balanced data
    X_moon, y_moon = make_moons(
        noise=0.3, random_state=0, n_samples=int(num_samples*2*imbalance_ratio))

    xy_moon = np.c_[X_moon, y_moon]

    df_moon = pd.DataFrame(xy_moon, columns=['x', 'y', 'class'])

    df_moon = df_moon.sort_values('class')
    df_moon = df_moon.iloc[0:num_samples, :]
    print("Moons created")
    print("Num in categories:")
    print(df_moon['class'].value_counts())
    return df_moon

# Make imbalanced

# Create circles


def create_circles(num_samples=NUM_SAMPLES, imbalance_ratio=IMBALANCE_RATIO):

    X_circles, y_circles = make_circles(
        noise=0.2, factor=0.5, random_state=1, n_samples=int(num_samples*2*imbalance_ratio))

    xy_circles = np.c_[X_circles, y_circles]

    df_circles = pd.DataFrame(xy_circles, columns=['x', 'y', 'class'])

    # Make imbalanced
    df_circles = df_circles.sort_values('class')
    df_circles = df_circles.iloc[0:num_samples, :]
    print("Circles created")
    print("Num in categories:")
    print(df_circles['class'].value_counts())
    return df_circles


def main():
    df = create_linearly_separable()
    df_moons = create_moons()
    df_circles = create_circles()

    if not os.path.exists("./csv/"):
        os.makedirs("./csv/")
    df.to_csv("./csv/df_linear.csv", index=False)
    df_moons.to_csv("./csv/df_moons.csv", index=False)
    df_circles.to_csv("./csv/df_circles.csv", index=False)
    print("All files create successfully!")


if __name__ == "__main__":
    main()
