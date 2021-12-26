"""
Use k-nearest neighbors to identify penguins.
The data is downloaded from
https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/master/palmerpenguins/data/penguins.csv
and the data set is metculously documented here
https://github.com/allisonhorst/palmerpenguins/blob/master/README.md

Data citation:
   Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer
   Archipelago (Antarctica) penguin data. R package version 0.1.0.
   https://allisonhorst.github.io/palmerpenguins/. doi:
   10.5281/zenodo.3960218.

License: CC0 Public Domain
"""
import os
import numpy as np

# Setting the random seed ensures that you'll get the same results each time
# you run this.
np.random.seed(645831)

# The number of most similar penguins that will be compared to
# determine each new penguin's type
global k

# The fraction of the data set that will be set aside to "train" the k-NN model
training_fraction = .5


def main():
    global k
    """
    Split the data into training and test sets. Use the training data
    to find the training penguins most similar to each test penguin.
    """
    features, labels = load_data()
    (train_features, train_labels, test_features, test_labels) = prep_data(
        features, labels)
    for x in range(1, 10):
        k = x
        n_test = test_labels.size
        top_score = 0
        for i_test in range(n_test):
            i_top = find_k_nearest(train_features, test_features[i_test, :])
            top_score+=score(train_labels[i_top], test_labels[i_test])
        print("Mean Score ", top_score/n_test, " for k = ", k)

def score(labels, actual):
    """
    Use the labels of the most similar neighbor penguins to make a prediction.
    Then compare the prediction to the actual.
    If the prediction was right, the model gets a 1.
    If the prediction was wrong, the model gets a 0.

    Arguments
    labels, 1D NumPy array of floats
    actual, float

    Returns
    credit, float
    """
    # Create three ballot boxes, one for each species.
    predictions = np.zeros(3)
    # Collect the votes from each neighbor.
    for label in labels:
        predictions[int(label)] += 1
    # Tally them up.
    max_vote = np.max(predictions)

    if (len(np.where(predictions == max_vote)[0]) > 1):
        if actual == any(np.where(predictions == max_vote)[0]):
            return 0.5
            
    predicted_label_index = np.where(predictions == max_vote)[0][0]
    
    # Was the prediction correct?
    if actual == predicted_label_index:
        return 1
    else:
        return 0


def load_data():
    """
    Open and parse the csv file containing the penguin data.

    Returns
    features, 2D NumPy array of floats
        Each row represents a penguin and each column a feature.
    labels, 1D NumPy array of floats
        There will be one element per penguin. The number will represent
        the type of penguin.
    """
    # data_filename = os.path.join("data", "penguins.csv")

    with open("penguins.csv", "rt") as f:
        data_lines = f.readlines()
        # The first row is full of column labels.
        # column_labels = data_lines[0].split(",")

        n_penguins = len(data_lines) - 1
        n_features = 5
        features = np.zeros((n_penguins, n_features))
        labels = np.zeros(n_penguins)

        # Describe how to convert sex and species text fields to numbers.
        sex_conversion = {"male": 0, "female": 1}
        label_conversion = {
            "Adelie": 0,
            "Chinstrap": 1,
            "Gentoo": 2,
        }

        # For each penguin, split the line up by commas, ignore
        # any residual whitespace on the ends, and pull out
        # the feature and label fields.
        # Start from row 1 so as to skip the column headings.
        # The try-except block catches the cases where the data is
        # missing and is replaced with and "NA". For now, we're choosing to
        # ignore all these data points.
        for i_penguin, line in enumerate(data_lines[1:]):
            line_data = line.split(",")
            try:
                numerical_data = [float(x.rstrip()) for x in line_data[2:6]]
                features[i_penguin, :4] = numerical_data
                # Convert sex and species to numbers.
                features[i_penguin, 4] = sex_conversion[line_data[6].rstrip()]
                labels[i_penguin] = label_conversion[line_data[0].rstrip()]
            except ValueError:
                # If any NA's are encountered in the numerical fields
                # just move along to the next penguin.
                pass
            except KeyError:
                # If any NA's are encountered in the sex conversion
                # just move along to the next penguin.
                pass

        return features, labels


def prep_data(features, labels):
    """
    Split the features and labels into training and testing groups.
    Shift and scale to have a mean of zero and a variance of 1.

    Adjust the global variable training_fraction to control how much
    information the algorithm has to work with. Lower values are more
    challenging.

    Arguments
    features, 2D NumPy array of floats
    labels, 1D NumPy array of floats

    Returns
    train_features, 2D NumPy array of floats
    train_labels, 1D NumPy array of floats
    test_features, 2D NumPy array of floats
    test_labels, 1D NumPy array of floats
    """
    n_penguins = labels.size
    n_train = int(n_penguins * training_fraction)
    # n_test = n_penguins - n_train

    # Divide up the data by generating a set of "straws"
    # of different length and distributing them randomly to all the penguins.
    # The penguins with the shortest straws are the training set.
    straws = np.arange(n_penguins)
    np.random.shuffle(straws)
    # The indices of the penguins in each set.
    i_train = straws[:n_train]
    i_test = straws[n_train:]

    # When normalizing, it's important that we only use the training
    # data to find the mean and standard deviation. If we were to use
    # the testing data, it could allow the algorithm to inadvertently
    # cheat a bit (a phenomenon called "data leakage").
    # But once we've found the mean and standard deviation of the training
    # data we're free to apply it to the testing data as a preprocessing
    # step. That doesn't break any rules.
    unscaled_train_features = features[i_train, :]
    unscaled_test_features = features[i_test, :]
    features_mean = np.mean(unscaled_train_features, axis=0)
    features_stddev = np.sqrt(np.var(unscaled_train_features, axis=0))
    train_features = (
        (unscaled_train_features - features_mean) /
        features_stddev)
    test_features = (
        (unscaled_test_features - features_mean) /
        features_stddev)

    train_labels = labels[i_train]
    test_labels = labels[i_test]

    return train_features, train_labels, test_features, test_labels



def find_k_nearest(train_points, test_point):
    """
    Find the distance between the a single test point and
    each of the training points.
    Use the Manhattan distance, the sum of differences in each feature.
    https://en.wikipedia.org/wiki/Taxicab_geometry
    Return the indices of the nearest neighbors.

    Arguments
    train_points, 2D NumPy array of floats
    test_point, 1D NumPy array of floats

    Returns
    i_top_k, 1D NumPy array of floats
    """
    distance = np.sum(np.abs(train_points - test_point[np.newaxis, :]), axis=1)
    order = np.argsort(distance)
    i_top_k = order[:k]
    return i_top_k


if __name__ == "__main__":
    main()
