"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
from scipy.stats import multivariate_normal

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    n_images = test.shape[0]
    gnb = GaussianBayes(train, train_labels)
    gnb.fit()
    # gnb = GaussianNB().fit(train, train_labels)
    knn = KNN(train, train_labels, 5, distanceMeasure=CosineEuclidDistance())
    return list(knn.predict(test))


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    # reduced_data = calculate_pca(data, 10)
    np.random.seed(42)
    reduced_data = gaussian_random_projection(data, 10)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)


def calculate_pca(data_input, num_of_features):
    row_means = np.mean(data_input, axis=0)
    centralized_data = data_input - row_means
    covariance_matrix = np.cov(centralized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvectors = eigenvectors[:, :num_of_features]
    return np.dot(eigenvectors.T, centralized_data.T).T


def gaussian_random_projection(data_input, num_of_features):
    current_features = data_input.shape[1]
    random_matrix = np.random.normal(np.zeros((num_of_features, current_features)), np.ones((num_of_features, current_features)) * (1 / num_of_features))
    return np.dot(random_matrix, data_input.T).T




class DistanceMeasure:
    def calculate(self, point1, point2, labels=None):
        raise Exception("Do not use the distance measure superclass!")


class EuclidianSquared(DistanceMeasure):
    def calculate(self, point1, point2, labels=None):
        return np.sum((point1 - point2) ** 2, axis=1)


class CosineDistance(DistanceMeasure):
    def calculate(self, point1, point2, labels=None):
        dot = np.dot(point1, point2)
        return 1 - (dot / (np.linalg.norm(dot)))


class CosineEuclidDistance(DistanceMeasure):
    def calculate(self, point1, point2, labels=None):
        euclidSquared = EuclidianSquared().calculate(point1, point2)
        cosineDistance = CosineDistance().calculate(point1, point2)
        return euclidSquared * (1 + cosineDistance)


class EuclidGaussianDistance(DistanceMeasure):
    def __init__(self, training_data, training_samples):
        self.gaussian = GaussianBayes(training_data, training_samples)
        self.gaussian.fit()

    def calculate(self, point1, point2, labels=None):
        if labels is None:
            raise Exception("Labels should be defined for the Euclidian Gaussian distance measure")
        euclidian = EuclidianSquared().calculate(point1, point2)
        probabilities = []
        for i in range(len(point1)):
            probabilities.append(1 - self.gaussian.calculate_probability(point2, labels[i]))
        return euclidian * probabilities


class KNN:
    def __init__(self, training_data_input, training_data_output, k, distanceMeasure: DistanceMeasure = EuclidianSquared()):
        self.training_data_input = training_data_input
        self.training_data_output = training_data_output
        self.k = k
        self.distanceMeasure = distanceMeasure
    def predict(self, data_input):
        predictions = list()
        for i in range(np.shape(data_input)[0]):
            distances = self.distanceMeasure.calculate(self.training_data_input, data_input[i,:], self.training_data_output)
            idx = distances.argsort()
            distances = distances[idx]
            labels = self.training_data_output[idx]
            k_labels = list(labels[:self.k].flatten())
            predictions.append(max(set(k_labels), key=k_labels.count))
        return predictions


class GaussianBayes:
    def __init__(self, training_data_input, training_data_output):
        self.training_data_input = training_data_input
        self.training_data_output = training_data_output

    def fit(self):
        labels = list(set(self.training_data_output.flatten()))
        self.labels = labels
        self.pdfs = []
        self.priors = []
        for label in labels:
            training_data_labelled = self.training_data_input[np.argwhere(self.training_data_output == label)].squeeze()
            mean = np.mean(training_data_labelled, axis=0)
            covariance = np.cov(training_data_labelled, rowvar=False)
            self.pdfs.append(multivariate_normal(mean=mean, cov=covariance))
            self.priors.append(len(training_data_labelled) / len(self.training_data_input))

    def calculate_probability(self, data, label):
        idx = self.labels.index(label)
        return self.pdfs[idx].pdf(data) * self.priors[idx]

    def predict(self, data_input):
        predictions = []
        for i in range(np.shape(data_input)[0]):
            probabilities = []
            for j in range(len(self.pdfs)):
                probabilities.append(self.calculate_probability(data_input[i], self.labels[j]))
            predictions.append(self.labels[probabilities.index(max(probabilities))])
        return predictions
