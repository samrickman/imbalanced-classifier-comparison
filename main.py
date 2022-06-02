from generation_and_classification import create_all_data, draw_classification_plot
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
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
from datetime import datetime

# Settings - define num samples
NUM_SAMPLES = 10_000
IMBALANCE_RATIO = 0.93
NOISE = 0.2  # non-linear - 0.25 is noisy, 0.02 not noisy.
CLASS_SEP = 1  # linear. 1 noisy, 3 not noisy.


# Define classifiers

classifiers = [
    LogisticRegression(class_weight='balanced', verbose=True),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    GaussianProcessClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
]

classifier_names = [
    "Logistic regression",
    "Random Forest",
    "Bagging",
    "Gradient Boosting",
    "Gaussian Process",
    "Neural Network",
]


# Data generating functions
# Code will work if you add make_blobs but not different to linearly separable
not_linearly_separable_fns = [make_circles, make_moons]

# Size of imbalanced groups for data generating functions
num_in_majority_group = int(NUM_SAMPLES*IMBALANCE_RATIO)
num_in_minority_group = int(NUM_SAMPLES * (1-IMBALANCE_RATIO)) + 1


def main():
    df_dict = create_all_data(
        not_linearly_separable_fns,
        num_samples=NUM_SAMPLES,
        imbalance_ratio=IMBALANCE_RATIO,
        num_in_majority_group=num_in_majority_group,
        num_in_minority_group=num_in_minority_group,
        noise=NOISE)

    draw_classification_plot(df_dict,
                             classifier_names,
                             classifiers,
                             outfile="algorithms-noisy"
                             )


if __name__ == "__main__":

    print(f'Script started at {datetime.now().strftime("%H:%M:%S")}')
    main()
    print(f'Script completed at {datetime.now().strftime("%H:%M:%S")}')
