"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

version: v2.0
"""
from typing import List

import numpy as np
from scipy.stats import multivariate_normal
from multiprocessing.pool import Pool
from collections import Counter

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Uses a basic KNN method using the EuclidianSquared distance measure

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    knn = KNN(train, train_labels, 5, distanceMeasure=EuclidianSquared())
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

    Uses PCA to reduce the data to N_DIMENSIONS
    Initially calculates the matrix of eigenvectors in process_training_data, and saves the resulting matrix in the model for later use

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    reduced_data = calculatePCA(data, np.array(model["pca_matrix"]))
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Calculates the matrix of eigenvectors for PCA usage
    Also computes a lookup table for p(chess piece | location on board), implemented using Bayes' theorem

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
    try:
        model["pca_matrix"]
    except KeyError:
        model["pca_matrix"] = calculatePCAMatrix(fvectors_train, N_DIMENSIONS).tolist()
    model["location_lookup_table"] = {}
    labels = list('.KkQqRrPpBbNn')
    for i in range(64):
        for label in labels:
            try:
                model["location_lookup_table"][label][i] = getPrior(label, i, labels_train)
            except KeyError:
                model["location_lookup_table"][label] = {}
                model["location_lookup_table"][label][i] = getPrior(label, i, labels_train)
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
    fvectors_train = np.array(model["fvectors_train"], dtype=np.float128)
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    Unlike classify_squares, this does not call classify. This is because a different distance measure is used
    Since we have access to the location on the board, this information can be used to make a better distance measure
    I call this distance measure EuclidWithPriors

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    fvectors_train = np.array(model["fvectors_train"], dtype=np.float128)
    labels_train = np.array(model["labels_train"])
    lookup_table = model["location_lookup_table"]
    knn = KNN(fvectors_train, labels_train, 5, distanceMeasure=EuclidWithPriors(lookup_table))
    return list(knn.predict(fvectors_test))


def getPrior(label: str, position: int, labels_train) -> float:
    """Computes p(chess piece | position on board)
    
    This is implemented using Bayes' theorem, so p(chess piece | position on board) = p(position on board | chess piece) * p(position on board) / p(chess piece)
    Since all positions on a chess board are equally likely, this is hardcoded as 1/64
    
    Args:
        label (str): The chess piece for which the probability is being calculated
        position (int): The position on the board, represented as a number from 0-63, 0 being the top left and 63 being the bottom right

    Returns:
        float: The probability of the chess piece given its location on the board
    """
    labels_flattened = labels_train.flatten()
    rawPrior = getRawPrior(label, labels_flattened)

    probabilityOfLocation = getProbabilityOfLocation(label, position, labels_flattened)

    # Bayes' theorem
    probabilityOfPiece = (probabilityOfLocation * (1 / 64)) / rawPrior
    return probabilityOfPiece


def getRawPrior(label: str, labels_flattened) -> float:
    """Calculates the 'raw prior' for a chess piece
    This is defined as just the probablity of the chess piece, p(chess piece)

    Args:
        label (str): The chess piece in question
        labels_flattened (np.ndarray): A 1D array containing all chess pieces in the training dataset

    Returns:
        float: The proportion of pieces in the dataset which match the given chess piece
    """
    labels_filtered = np.where(labels_flattened == label)[0]
    return len(labels_filtered) / len(labels_flattened)

def getProbabilityOfLocation(label: str, position: int, labels_flattened) -> float:
    """Calculates the probability of a location given a chess piece, p(location | chess piece)
    This is defined as the proportion of chess squares of that location which contain the chess piece in question

    Args:
        label (str): The chess piece in question
        position (int): The position on the chess board, from 0 to 63

    Returns:
        float: The probability of that location given the chess board
    """
    labels_with_locations = []
    idxs = np.arange(position, len(labels_flattened), 64)
    labels_with_locations = labels_flattened[idxs]

    labels_with_locations_filtered = np.where(labels_with_locations == label)[0]
    return len(labels_with_locations_filtered) / len(labels_with_locations)


def calculatePCAMatrix(data_input, num_of_features: int):
    """Calculates the matrix of eigenvectors for some data
    
    Args:
        data_input (np.ndarray): The data for which the matrix is to be calculated
        num_of_features (int): The number of features after the matrix is applied on the data

    Returns:
        np.ndarray: The matrix of eigenvectors for the data. This matrix contains num_of_features eigenvectors
    """
    covariance_matrix = np.cov(data_input, rowvar=False)
    # eigh can be used as an optimization here, as the covariance matrix is symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvectors = eigenvectors[:, :num_of_features]
    return eigenvectors


def calculatePCA(data_input, matrix):
    """Calculates the reduced data given a data input and a matrix from calculatePCAMatrix
    
    Args:
        data_input (np.ndarray): The data input to be reduced
        matrix (np.ndarray): The relevant matrix of eigenvectors from calculatePCAMatrix

    Returns:
        np.ndarray: The transformed data_input, reduced to the desired number of features
    """
    return np.dot(matrix.T, data_input.T).T


def gaussianRandomProjection(data_input, num_of_features: int):
    """Carries out gaussian random projection on some data
    
    Args:
        data_input (np.ndarray): The data input to be transformed
        num_of_features (int): The desired number of features

    Returns:
        np.ndarray: The input data, having been reduced to num_of_features via gaussian random projection
    """
    current_features = data_input.shape[1]
    random_matrix = np.random.normal(np.zeros((num_of_features, current_features)), np.ones((num_of_features, current_features)) * (1 / num_of_features))
    return np.dot(random_matrix, data_input.T).T




# A superclass for distance measures, this should never be directly instantiated
class DistanceMeasure:
    def calculate(self, point1, point2, labels=None, position=None):
        raise Exception("Do not use the distance measure superclass!")


# The Euclidian squared distance measure
# While the Euclidian measure would take the square root, the Euclidian squared distance measure does not. This produces an identical distance measure,
# but saves on CPU time
class EuclidianSquared(DistanceMeasure):
    def calculate(self, point1, point2, labels=None, position=None):
        return np.sum((point1 - point2) ** 2, axis=1)


class EuclidDistance(DistanceMeasure):
    def calculate(self, point1, point2, labels=None, position=None):
        return np.sqrt(EuclidianSquared().calculate(point1, point2))


class LpNormDistance(DistanceMeasure):
    def __init__(self, p):
        self.p = p

    def calculate(self, point1, point2, labels=None, position=None):
        return np.power(np.abs(np.sum((point1 - point2) ** self.p, axis=1)), 1 / self.p)


class Manhattan(DistanceMeasure):
    def calculate(self, point1, point2, labels=None, position=None):
        return np.sum(np.abs(point1 - point2), axis=1)


# The cosine distance measure
class CosineDistance(DistanceMeasure):
    def calculate(self, point1, point2, labels=None, position=None):
        dot = np.dot(point1, point2)
        return 1 - (dot / (np.linalg.norm(dot)))


# A combined distance measure of Euclidian squared and Cosine, defined as the Euclidian squared measure * (1 + Cosine distance)
class CosineEuclidDistance(DistanceMeasure):
    def calculate(self, point1, point2, labels=None, position=None):
        euclidSquared = EuclidianSquared().calculate(point1, point2)
        cosineDistance = CosineDistance().calculate(point1, point2)
        return euclidSquared * (1 + cosineDistance)


# A distance measure which attempts to combine KNN and a Gaussian model of the data.
class EuclidGaussianDistance(DistanceMeasure):
    # The idea is to fit a Gaussian model of the data, and use this to scale Euclidian squared distances
    # In practice, this is only as affective as a Gaussian model using the Bayes classifier
    def __init__(self, training_data, training_samples):
        self.gaussian = GaussianBayes(training_data, training_samples)
        self.gaussian.fit()
        self.training_samples = training_samples

    def calculate(self, point1, point2, labels, position=None):
        euclidian = EuclidianSquared().calculate(point1, point2)
        probabilities = []
        for i in range(len(point1)):
            probabilities.append(1 - self.gaussian.calculate_probability(point2, labels[i]))
        return euclidian * probabilities


# Another non-standard distance measure
# This one uses the priors instead of a Gaussian model to scale the distances
# With standard data, the priors are simply the proportions of chess pieces and so this simply causes the more popular labels to be selected
# When board data is included, though, this allows the classifier to model the fact that some chess pieces should not be in certain places
# It should be noted that this is not a true model, though, and is just simple statistics put in practice
class EuclidWithPriors(DistanceMeasure):
    def __init__(self, lookup_table):
        self.lookup_table = lookup_table

    def calculate(self, point1, point2, labels, position):
        euclidDistance = LpNormDistance(2).calculate(point1, point2)
        # priors = getPriors(labels, positions, self.training_data, self.training_samples) + 1
        priors = []
        for label in labels:
            priors.append(self.lookup_table[label][str(position)] + 1)
        return (euclidDistance / np.array(priors, dtype=np.float128))


# The K-Nearest Neighbors classifier
class KNN:
    def __init__(self, training_data_input, training_data_output, k, distanceMeasure: DistanceMeasure = EuclidianSquared()):
        self.training_data_input = training_data_input
        self.training_data_output = training_data_output
        self.k = k
        self.distanceMeasure = distanceMeasure

    # This is a multi-threaded (technically multi-process) predict, as some distance measures are very time-heavy
    def predict(self, data_input):
        NUM_THREADS = 10
        pools = []
        results = []
        step_size = np.shape(data_input)[0] // NUM_THREADS
        for i in range(NUM_THREADS-1):
            start = i * step_size
            end = (i + 1) * step_size
            pools.append(Pool(processes=1))
            results.append(pools[-1].apply_async(self.sub_predict, (data_input, start, end, )))
        pools.append(Pool(processes=1))
        results.append(pools[-1].apply_async(self.sub_predict, (data_input, (step_size * (NUM_THREADS - 1)), np.shape(data_input)[0])))
        predictions = []
        for result in results:
            predictions.append(result.get())
        return np.array(predictions).flatten()

    # This is where the actual computations are done, and what the predict() method calls in its threads
    def sub_predict(self, data_input, start, end):
        predictions = list()
        for i in range(start, end):
            distances = self.distanceMeasure.calculate(self.training_data_input, data_input[i,:], self.training_data_output, i % 64)
            idx = np.argsort(distances)[:self.k]
            labels = self.training_data_output[idx]
            k_labels = list(labels[:self.k].flatten())
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return predictions


# This is a Bayesian classifier which models the data using a Gaussian distribution
# Though this is more complex than the KNN classifier, in practice it is less effective
class GaussianBayes:
    def __init__(self, training_data_input, training_data_output, priors=None):
        self.training_data_input = training_data_input
        self.training_data_output = training_data_output
        self.priors = priors

    def fit(self):
        labels = list(set(self.training_data_output.flatten()))
        self.labels = labels
        self.pdfs = []
        doPriors = False
        if self.priors is None:
            doPriors = True
        self.priors = {}
        for label in labels:
            training_data_labelled = self.training_data_input[np.argwhere(self.training_data_output == label)].squeeze()
            mean = np.mean(training_data_labelled, axis=0)
            covariance = np.cov(training_data_labelled, rowvar=False)
            self.pdfs.append(multivariate_normal(mean=mean, cov=covariance))
            if doPriors:
                self.priors[label] = len(training_data_labelled) / len(self.training_data_input)

    def calculate_probability(self, data, label):
        idx = self.labels.index(label)
        return self.pdfs[idx].pdf(data) * self.priors[label]

    def predict(self, data_input):
        predictions = []
        for i in range(np.shape(data_input)[0]):
            probabilities = []
            for j in range(len(self.pdfs)):
                probabilities.append(self.calculate_probability(data_input[i], self.labels[j]))
            predictions.append(self.labels[probabilities.index(max(probabilities))])
        return predictions
