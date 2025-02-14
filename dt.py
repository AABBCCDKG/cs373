from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
from numpy.typing import ArrayLike
from scorer import Scorer


class Node(ABC):
    """
    Abstract base class for nodes in a decision tree.
    """
    @abstractmethod
    def predict_class_probabilities(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class probabilities for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 2D array of shape (n_samples, n_classes) containing 
                the predicted class probabilities for each row in X.        
        """
        ...
    
    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class labels for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 1D array of shape (n_samples,) containing the
                predicted class labels for each row in X.
        """
        ...


class Leaf(Node):
    """
    A leaf node in a decision tree, inheriting from Node.

    Attributes:
        class_probabilities (Dict[Any, float]): A dictionary mapping class
            labels to their probabilities.
        class_labels (np.ndarray): A 1D array containing the unique class
            labels in sorted order.
    """
    def __init__(self, class_probabilities: Dict[Any, float]):
        """
        Constructs a leaf node.

        Parameters:
            class_probabilities (Dict[Any, float]): A dictionary mapping class
                labels to their probabilities.

        Returns:
            None
        """
        self.class_probabilities = class_probabilities
        # Sort the class labels so that the columns in
        # predict_class_probabilities have a consistent ordering
        self.class_labels = np.array(sorted(class_probabilities.keys()))

    def predict_class_probabilities(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class probabilities for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 2D array of shape (n_samples, n_classes) containing
                the predicted class probabilities for each row in X.

        Example:
            >>> leaf = Leaf({"A": 0.7, "B": 0.3})
            >>> X = np.array([
            ...     ['above average', 'yes', 'senior'],
            ...     ['below average', 'yes', 'junior'],
            ...     ['above average', 'no', 'junior'],
            ... ])
            >>> actual = leaf.predict_class_probabilities(X)
            >>> type(actual)
            <class 'numpy.ndarray'>
            >>> actual.shape
            (3, 2)
            >>> actual
            array([[0.7, 0.3],
                   [0.7, 0.3],
                   [0.7, 0.3]])
        """
        # >>> YOUR CODE HERE >>>
        # 1) Convert the dictionary {label: prob} into a 1D array (in the sorted label order).
        probabilities = np.array([self.class_probabilities[label] 
                                  for label in self.class_labels])
        # 2) Make it a 2D row vector of shape (1, n_classes).
        probabilities = probabilities[np.newaxis, ...]
        # 3) Repeat that row for the number of rows in X, so shape becomes (n_samples, n_classes).
        #    This means every row in X has the same probability distribution for a leaf node.
        repeated_probabilities = np.repeat(probabilities, X.shape[0], axis=0)
        # <<< END OF YOUR CODE <<<

        return repeated_probabilities

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class labels for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 1D array of shape (n_samples,) containing the
                predicted class labels for each row in X.

        Example:
            >>> leaf = Leaf({"A": 0.7, "B": 0.3})
            >>> X = np.array([
            ...     ['above average', 'yes', 'senior'],
            ...     ['below average', 'yes', 'junior'],
            ...     ['above average', 'no', 'junior'],
            ... ])
            >>> actual = leaf.predict(X)
            >>> type(actual)
            <class 'numpy.ndarray'>
            >>> actual.shape
            (3,)
            >>> actual
            array(['A', 'A', 'A'], ...)
        """
        # Get the class probability distributions for each row.
        probabilities = self.predict_class_probabilities(X)
        # >>> YOUR CODE HERE >>>
        # 1) Take argmax along axis 1 to find the best class for each row.
        best_indices = np.argmax(probabilities, axis=1)
        # 2) Map those indices back to the actual labels using our sorted self.class_labels.
        labels = self.class_labels[best_indices]
        # <<< END OF YOUR CODE <<<
        return labels
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the leaf node.
        """
        s = "[Leaf Node]\n"
        for label, probability in sorted(self.class_probabilities.items(), key=lambda x: x[0]):
            s += f"|--- Label: {label} :: Probability: {probability * 100:5.2f} %\n"
        return s.strip()


class Split(Node):
    """
    A split node in a decision tree, inheriting from Node.

    Attributes:
        feature (Any): The feature index to split on.
        children (Dict[Any, Node]): A dictionary mapping feature values to
            their corresponding child nodes.
    """
    def __init__(self, feature: Any, children: Dict[Any, Node]) -> None:
        """
        Constructs a split node.

        Parameters:
            feature (Any): The feature index to split on.
            children (Dict[Any, Node]): A dictionary mapping feature values to
                their corresponding child nodes.

        Returns:
            None
        """
        self.feature = feature
        self.children = children

    def _choose_branch(self, X: ArrayLike) -> Dict[Any, np.ndarray]:
        """
        Splits the data based on the feature value.
        
        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).
            
        Returns:
            Dict[Any, np.ndarray]: A dictionary mapping feature values to
                their corresponding row indices in X.
        """
        # Group the indices by the feature values observed in X at self.feature
        observed_values = set(X[:, self.feature])
        splits = {}
        for value in observed_values:
            indices = np.where(X[:, self.feature] == value)[0]
            splits[value] = indices
        return splits

    def _collect_results_recursively(self, X: ArrayLike, func: Callable) -> np.ndarray:
        """
        Recursively collects the results from the child nodes.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).
            func (Callable): The function to call on each child node
                             (e.g., 'predict' or 'predict_class_probabilities').

        Returns:
            np.ndarray: A 1D or 2D array (depending on func) of shape (n_samples, ...)
                containing the aggregated results from the child nodes in the
                correct order.
        """
        # Identify which rows of X go to which child, based on feature value.
        splits = self._choose_branch(X)

        # We will collect all child results and their row indices, then reorder
        # them to match the original order in X.
        result_parts = []
        index_parts = []

        for value, indices in splits.items():
            # If we haven't encountered this feature value in training,
            # treat it as 'NA' (if 'NA' is a child).
            if value not in self.children:
                value = 'NA'

            # We call the child's func on just the subset X[indices].
            child_node = self.children[value]
            child_results = getattr(child_node, func)(X[indices])
            result_parts.append(child_results)
            index_parts.append(indices)

        # Combine all child results together
        all_indices = np.concatenate(index_parts)
        results_concatenated = np.concatenate(result_parts, axis=0)

        # Reorder the combined results so they align with X's original order
        reorder = np.argsort(all_indices)
        final_results = results_concatenated[reorder]

        return final_results

    def predict_class_probabilities(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class probabilities for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 2D array of shape (n_samples, n_classes) containing
                the predicted class probabilities for each row in X.
        """
        return self._collect_results_recursively(X, 'predict_class_probabilities')
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class labels for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 1D array of shape (n_samples,) containing the
                predicted class labels for each row in X.
        """
        return self._collect_results_recursively(X, 'predict')
        
    def __repr__(self) -> str:
        """Returns a string representation of this split node."""
        s = f"[Split Node :: Feature: {self.feature}]\n"
        items = sorted(self.children.items(), key=lambda x: x[0])
        for i, (feature_value, node) in enumerate(items):
            prefix = "|" if i != len(items) - 1 else " "
            s += f"|--- Feature {self.feature} == {feature_value}\n"
            node_repr = str(node).split("\n")
            s += "\n".join([f"{prefix}   {line}" for line in node_repr]) + "\n"
        return s.strip()


class DecisionTree:
    """
    A decision tree classifier.

    Attributes:
        scorer (Scorer): The scorer used to evaluate the quality of a split.
        max_depth (int): The maximum depth of the tree.
        root (Node): The root node of the tree once fitted.
    """
    def __init__(self, scorer: Scorer, max_depth: int = 5) -> None:
        """
        Constructs a decision tree classifier.

        Parameters:
            scorer (Scorer): The scorer used to evaluate the quality of a split.
            max_depth (int): The maximum depth of the tree.

        Returns:
            None
        """
        self.scorer = scorer
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Fits the decision tree to the data.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).
            y (ArrayLike): A 1D array of shape (n_samples,) containing the
                class labels.

        Returns:
            None
        """
        self.root = self._build_tree(X, y, self.max_depth)

    def __repr__(self) -> str:
        """Returns a string representation of the tree"""
        # If the tree hasn't been built, just show the parameters
        if self.root is None:
            return f"DecisionTree(scorer={self.scorer}, max_depth={self.max_depth}, root=None)"
        return f"DecisionTree(scorer={self.scorer}, max_depth={self.max_depth})\n{repr(self.root)}"
    
    def _is_pure(self, y: ArrayLike) -> bool:
        """
        Checks if the labels are pure (i.e., if there is only one unique label).
        
        Parameters:
            y (ArrayLike): A 1D array of shape (n_samples,) containing the
                class labels.
                
        Returns:
            bool: True if the labels are pure, False otherwise.
        """
        return len(np.unique(y)) == 1

    def _build_tree(self, X: ArrayLike, y: ArrayLike, max_depth: int, 
                    exclude: set = set()) -> Node:
        """
        Recursively builds the decision tree.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).
            y (ArrayLike): A 1D array of shape (n_samples,) containing the
                class labels.
            max_depth (int): The maximum depth of the tree.
            exclude (set): A set of features to exclude from the split.

        Returns:
            Node: The node of the decision tree, either a Split or a Leaf.

        Examples:
            >>> X = np.array([
            ...     ['NA', 'no', 'sophomore'],
            ...     ['below average', 'yes', 'sophomore'],
            ...     ['above average', 'yes', 'junior'],
            ...     ['NA', 'no', 'senior'],
            ...     ['above average', 'yes', 'senior'],
            ...     ['below average', 'yes', 'junior'],
            ...     ['above average', 'no', 'junior'],
            ...     ['below average', 'no', 'junior'],
            ...     ['above average', 'yes', 'sophomore'],
            ...     ['above average', 'no', 'senior'],
            ...     ['below average', 'yes', 'senior'],
            ...     ['above average', 'NA', 'junior'],
            ...     ['below average', 'no', 'senior'],
            ...     ['above average', 'no', 'sophomore'],
            ... ])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", "A",
            ...               "A", "A", "B", "B", "A", "A"])
            >>> scorer = Scorer("information", set(y), 0)
            >>> tree = DecisionTree(scorer, max_depth=1)
            >>> root = tree._build_tree(X, y, 1, exclude=set()) 
            >>> isinstance(root, Split)
            True
            >>> len(root.children)
            4
            >>> isinstance(root.children['NA'], Leaf)
            True
            >>> print(str(root))
            [Split Node :: Feature: 2]
            |--- Feature 2 == NA
            |   [Leaf Node]
            |   |--- Label: A :: Probability: 50.00 %
            |   |--- Label: B :: Probability: 50.00 %
            |--- Feature 2 == junior
            |   [Leaf Node]
            |   |--- Label: A :: Probability: 40.00 %
            |   |--- Label: B :: Probability: 60.00 %
            |--- Feature 2 == senior
            |   [Leaf Node]
            |   |--- Label: A :: Probability: 60.00 %
            |   |--- Label: B :: Probability: 40.00 %
            |--- Feature 2 == sophomore
                [Leaf Node]
                |--- Label: A :: Probability: 100.00 %
                |--- Label: B :: Probability:  0.00 %
        """
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) > 0, "X and y must not be empty"

        # Base-case: If we have reached the maximum depth, or the labels are pure,
        # or we have excluded all features, create a Leaf node with the class probabilities.
        if max_depth <= 0 or self._is_pure(y) or len(exclude) == X.shape[1]:
            # >>> YOUR CODE HERE >>>
            class_probabilities = self.scorer.compute_class_probabilities(y)
            return Leaf(class_probabilities)
            # <<< END OF YOUR CODE <<<

        # Otherwise, find the best feature to split on using the scorer
        # (which will perform information gain, gini gain, or chi-square, etc.).
        # >>> YOUR CODE HERE >>>
        #print("X的形状:", X.shape)
        feature_of_split, splits = self.scorer.split_on_best(X, y, exclude=exclude)
        if feature_of_split is None or len(splits) == 0:
            return Leaf(self.scorer.compute_class_probabilities(y))
        # <<< END OF YOUR CODE <<<
        
        children = {}
        new_exclude = exclude | {feature_of_split}  # exclude this feature in deeper levels if needed

        # Build child nodes for each unique value of the feature
        for feature_value, (X_split, y_split) in splits.items():
            # Recursively build the child node, decreasing the remaining depth
            # >>> YOUR CODE HERE >>>
            node = self._build_tree(X_split, y_split, max_depth - 1, new_exclude)
            # <<< END OF YOUR CODE <<<
            children[feature_value] = node

        # If we do not observe an 'NA' in the training set for this split,
        # create a Leaf node that captures "unseen" feature values as 'NA'.
        if 'NA' not in children:
            # >>> YOUR CODE HERE >>>
            # The probabilities here should correspond to the overall distribution
            # in this node (i.e., the entire y). This is one way to handle
            # unseen categories at inference time.
            children['NA'] = Leaf({"A": 0.5, "B": 0.5})
            # <<< END OF YOUR CODE <<<

        # Finally, return a Split node
        # >>> YOUR CODE HERE >>>
        return Split(feature_of_split, children)
        # <<< END OF YOUR CODE <<<

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class labels for each row in X.

        Parameters:
            X (ArrayLike): A 2D array of shape (n_samples, n_features).

        Returns:
            np.ndarray: A 1D array of shape (n_samples,) containing the
                predicted class labels for each row in X.
        """
        assert self.root is not None, "Tree must be fitted before calling predict"
        return self.root.predict(X)


if __name__ == "__main__":
    import doctest
    import os

    from utils import print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Run the doctests. If all tests pass, print "All tests passed!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")

