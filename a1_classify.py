#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

all_classifiers = [SGDClassifier(),
                   GaussianNB(),
                   RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05),
                   AdaBoostClassifier()
                   ]


def accuracy(C):
    """ Compute accuracy given Numpy array confusion matrix C. Returns a floating point value """
    # the sum of diagonal elements divide the sum of all
    if np.sum(C) == 0:
        return 0
    else:
        return np.trace(C)/np.sum(C)


def recall(C):
    """ Compute recall given Numpy array confusion matrix C. Returns a list of floating point values """
    # divided by the total num of the sum of all the rows for this column
    temp_list = []
    assert len(C) == 4
    for i in range(4):
        temp_list.append(C[i,i]/np.sum(C[i,:]))

    return temp_list


def precision(C):
    """ Compute precision given Numpy array confusion matrix C. Returns a list of floating point values """
    # divided by the total num of the sum of all the columns for this row
    temp_list = []
    assert len(C) == 4
    for i in range(4):
        temp_list.append(C[i,i]/np.sum(C[:, i]))
    return temp_list


def class31(output_dir, X_train, X_test, y_train, y_test):
    """ This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    """
    best_acc = 0
    best_index = 0

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i, classifier in enumerate(all_classifiers):
            # clone the function to make sure it's clear before the training start
            classifier_clone = clone(classifier)

            # train and get the output
            classifier_clone.fit(X_train, y_train)
            conf_matrix = confusion_matrix(y_test, classifier_clone.predict(X_test))

            # sort the output data
            classifier_name = str(classifier_clone).split("(")[0]
            print("Now running: 3.1 " + classifier_name)
            accuracy_var = accuracy(conf_matrix)
            recall_var = recall(conf_matrix)
            precision_var = precision(conf_matrix)

            if accuracy_var > best_acc:
                best_index = i
                best_acc = accuracy_var


            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {accuracy_var:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall_var]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision_var]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

        # analysis
        outf.write(f'under my environment, the ranking is:\n')
        outf.write(f'AdaBoost >> RandomForest > MLP >> SGD > Gaussian\n')

    return best_index


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    """ This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   """

    # mostly the iBest should be 4
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        x_1k = None
        y_1k = None
        for i in range(5):
            if i == 0:
                size = 1000
            else:
                size = i * 5000

            x_temp = X_train[:size]
            y_temp = y_train[:size]

            print(f"Now running: 3.2 size: {size}\n")

            if i == 0:
                x_1k = x_temp
                y_1k = y_temp

            classifier_clone = clone(all_classifiers[iBest])
            classifier_clone.fit(x_temp, y_temp)
            conf_matrix = confusion_matrix(y_test, classifier_clone.predict(X_test))
            accuracy_var = accuracy(conf_matrix)
            # For each number of training examples, compute results and write
            # the following output:
            outf.write(f'{size}: {accuracy_var:.4f}\n')
        outf.write(f'As you can see the accuracy increase steadily.\n')
        outf.write(f'Program perform better after is get more data to learn and practice\n')
    return (x_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    """ This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    """
    print("Now running: 3.3")

    k_set = [5, 50]
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Part 1
        selector_5 = SelectKBest(f_classif, k_set[0])
        selector_50 = SelectKBest(f_classif, k_set[1])

        x_5_full = selector_5.fit_transform(X_train, y_train)
        features_5_full = selector_5.get_support(True)
        x_50_full = selector_50.fit_transform(X_train, y_train)
        features_50_full = selector_50.get_support(True)

        p_5_full = selector_5.pvalues_
        p_list_5 = [p_5_full[index] for index in features_5_full]
        p_50_full = selector_50.pvalues_
        p_list_50 = [p_50_full[index] for index in features_50_full]

        outf.write(f'5 p-values: {[format(pval) for pval in p_list_5]}\n')
        outf.write(f'50 p-values: {[format(pval) for pval in p_list_50]}\n')

        # Part 2
        # idk if fit_transform is erasing previous attempt or just improving based on the previous attempt, so i just start a new one
        # selector_5 = SelectKBest(f_classif, k_set[0])

        x_5_1k = selector_5.fit_transform(X_1k, y_1k)
        features_5_1k = selector_5.get_support(True)
        p_5_1k = selector_5.pvalues_
        p_list_5_1k = [p_5_1k[index] for index in features_5_1k]

        print("k = 5, size = full")
        print(features_5_full)
        print(p_list_5)

        print("k = 5, size = 1k")
        print(features_5_1k)
        print(p_list_5_1k)

        classifier_5_full = clone(all_classifiers[i])
        classifier_5_full.fit(x_5_full, y_train)
        con_matrix_5_full = confusion_matrix(y_test, classifier_5_full.predict(selector_5.transform(X_test)))
        acc_5_full = accuracy(con_matrix_5_full)

        classifier_5_1k = clone(all_classifiers[i])
        classifier_5_1k.fit(x_5_1k, y_1k)
        con_matrix_5_1k = confusion_matrix(y_test, classifier_5_1k.predict(selector_5.transform(X_test)))
        acc_5_1k = accuracy(con_matrix_5_1k)

        outf.write(f'Accuracy for 1k: {acc_5_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {acc_5_full:.4f}\n')

        # Part 3 and 4
        feature_intersection = np.intersect1d(features_5_full, features_5_1k)
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {features_5_full}\n')

        # answers
        outf.write(
            f'(a): according to the output on my environment, the most important features share by two set is:\n')
        outf.write(f'Feature 6 -> Number of past-tense verbs; and other two features in LIWC package\n')
        outf.write(f'I believe it is because more past-tense verbs means the writer concern more on the history and\n')
        outf.write(f'can some how represent a person\'s personality.\n')

        outf.write(f'(b): The p-value decrease a lot when the size of train set increase.\n')
        outf.write(
            f'I think it is because, although we have got the 5 best features, but it is still not sufficient to\n')
        outf.write(
            f'only rely on those 5 features, and when the data size increase, our reliance on those 5 features\n')
        outf.write(f'decrease, so the p value get smaller. You can find the comparison in terminal output.\n')

        outf.write(f'(c): Same as (a), there is only 1 feature: Feature 6 I can analyze. And the analysis is same as\n')
        outf.write(f'it in (a), all other 4 are in LIWC package and I don\'t know what is it\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    """ This function performs experiment 3.4
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
    """
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        acc_data = np.zeros((5,5))
        x_total = np.concatenate([X_train, X_test], axis=0)
        y_total = np.concatenate([y_train, y_test], axis=0)
        divider = KFold(shuffle=True)

        for classifier_index in range(5):
            # classifier_index is the column index
            classifier = all_classifiers[classifier_index]
            classifier_name = str(classifier).split("(")[0]
            fold_index = 0  # fold_index is the row index in data matrix
            for train_index, test_index in divider.split(x_total):
                classifier_clone = clone(classifier)
                x_train, x_test = x_total[train_index], x_total[test_index]
                y_train, y_test = y_total[train_index], y_total[test_index]

                classifier_clone.fit(x_train, y_train)
                con_matrix = confusion_matrix(y_test, classifier_clone.predict(x_test))
                acc = accuracy(con_matrix)
                acc_data[fold_index, classifier_index] = acc
                fold_index += 1
            # so for now one row of the 5*5 matrix should be filled
        # for now the whole matrix should have been completed.

        print(acc_data)

        for row in acc_data:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in row]}\n')

        p_list = []
        compare_list = list(range(5))
        compare_list.remove(i)
        print(compare_list)

        for column_index_to_compare in compare_list:
            S = ttest_rel(acc_data[:, column_index_to_compare], acc_data[:, i])
            p_list.append(S.pvalue)

        outf.write(f'p-values: {[round(pval, 4) for pval in p_list]}\n')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    npzFile = np.load(args.input)
    input_array = npzFile[npzFile.files[0]]
    _x_train, _x_test, _y_train, _y_test = train_test_split(input_array[:, :-1], input_array[:, -1], test_size=0.2)

    best_classifier_index = class31(args.output_dir, _x_train, _x_test, _y_train, _y_test)

    x_1k, y_1k = class32(args.output_dir, _x_train, _x_test, _y_train, _y_test, iBest=best_classifier_index)

    class33(args.output_dir, _x_train, _x_test, _y_train, _y_test, i=best_classifier_index, X_1k=x_1k, y_1k=y_1k)

    class34(args.output_dir, _x_train, _x_test, _y_train, _y_test, i=best_classifier_index)