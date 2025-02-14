import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    print("tqdm not found. Install it to see the progress bar if you want.")
    tqdm = lambda x: x

class KNearestNeighbor:
    """
    This class implements the KNN classifier.
    
    Attributes:
        k (int): The number of nearest neighbors to consider.
    """
    
    def __init__(self, k: int) -> None:
        """
        The constructor for KNearestNeighbor class. Saves the k value to
        `self.k`. Initializes the training data and labels to `None`.

        Parameters:
            k (int): The number of nearest neighbors to consider.

        Returns:
            None

        Examples:
            >>> knn = KNearestNeighbor(3)
            >>> knn.k
            3
            >>> knn.X_train
            >>> knn.y_train
        """

        # >>> YOUR CODE HERE >>>
        self.k = k
        self.X_train = None
        self.y_train = None
        # <<< END OF YOUR CODE <<<

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        This function saves the training data and labels to `self.X_train` and
        `self.y_train` respectively.

        Parameters:
            X (np.ndarray): The training data.
            y (np.ndarray): The labels for the training data.

        Returns:
            None

        Examples:
            >>> knn = KNearestNeighbor(1)
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> knn.fit(X, y)
            >>> knn.X_train
            array([[1, 2],
                   [3, 4]])
            >>> knn.y_train
            array([0, 1])
        """

        assert X_train.shape[0] == y_train.shape[0], \
            "The number of training data and labels must be the same."

        assert self.k <= X_train.shape[0], "The number of nearest neighbors cannot be greater than the number of training data." # type: ignore

        # >>> YOUR CODE HERE >>>
        self.X_train = X_train
        self.y_train = y_train
        # <<< END OF YOUR CODE <<<


    def calc_distance(self, x, p=3):
        """
        This function calculates the Minkowski distance of order p between the training
        data and the data to predict.

        Parameters:
        x (np.ndarray): The test data.
        p (int): The order of the norm. Default is 2 (Euclidean distance).

        Returns:
        distance (np.ndarray): The Minkowski distance between the training data and the test data.
        """

        # >>> YOUR CODE HERE >>>
        differences = np.abs(self.X_train - x)
        powered_differences = differences ** p
        sum_of_powers = np.sum(powered_differences, axis = 1)
        distances = sum_of_powers ** (1/p)
        # <<< END OF YOUR CODE <<<
        return distances
    
    def get_top_k(self, distance: np.ndarray) -> np.ndarray:
        """
        This function returns the indices of the top k smallest distances.

        You may find `np.argsort` useful.

        Parameters:
            distance (np.ndarray): The Manhattan distance between the training 
            data and the test data.

        Returns:
            top_k (np.ndarray): The indices of the top k smallest distances.

        Examples:
            >>> knn = KNearestNeighbor(1)
            >>> distance = np.array([1, 2, 3, 4, 5])
            >>> knn.get_top_k(distance)
            array([0]...)
        """

        # >>> YOUR CODE HERE >>>
        top_k = np.argsort(distance)[:self.k]
        # <<< END OF YOUR CODE <<<

        return top_k
    
    def predict(self, X_predict: np.ndarray, p: int) -> np.ndarray:
        """
        This function predicts the labels for the test data by calculating the
        Manhattan distance between the training data and the test data, and
        finding the top k smallest distances. The predicted label is the label
        that appears the most in the top k smallest distances.

        You may find `np.argmax` and `np.bincount` useful.

        Parameters:
            X_predict (np.ndarray): The test data.
            p (int): The order of the norm used for the Minkowski distance calculation.

        Returns:
            y_predict (np.ndarray): The predicted labels for the test data.

        Examples:
            >>> knn = KNearestNeighbor(1)
            >>> X_train = np.array([[1, 2], [3, 4]])
            >>> y_train = np.array([0, 1])
            >>> X_predict = np.array([[5, 6]])
            >>> knn.fit(X_train, y_train)
            >>> knn.predict(X_predict, 3)
            array([1])
        """

        assert self.X_train is not None and self.y_train is not None, \
            "The training data and labels must be initialized."
        
        assert X_predict.shape[1] == self.X_train.shape[1], "The number of features for the training data and test data must be the same." # type: ignore

        # Initialize the predicted labels to be all zeros.
        y_predict = np.zeros(X_predict.shape[0], dtype=self.y_train.dtype)

        # >>> YOUR CODE HERE >>>
        for i in tqdm(range(X_predict.shape[0])):
            distances = self.calc_distance(X_predict[i], p)
            top_k_indices = self.get_top_k(distances)
            top_k_labels = self.y_train[top_k_indices]
            
            y_predict[i] = np.argmax(np.bincount(top_k_labels))
    
        # <<< END OF YOUR CODE <<<


        return y_predict
    
if __name__ == "__main__":
    import doctest
    import os

    from utils import print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Run the doctests. If all tests pass, print "All tests pass!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
        
        
    
