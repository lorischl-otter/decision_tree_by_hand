class DecisionTreeClassifier:
    """
    Manual implementation of a Decision Tree Classifier
    for numeric and/or ranked data, using the metric of gini impurity.

    See example.py for sample implementation code.


    Instantiation Parameters
    ----------
    max_depth: int, default = None
    Similar to the sklearn implementation, max_depth provides a parameter
    to prevent the tree from overfitting. It is the maximum depth of the tree.

    min_samples_split: int, default = 2
    Similar to the sklearn implementation, min_samples_split provides the
    minimum number of observations in a given leaf in order for
    the leaf to split and become a parent node.


    Methods
    ----------
    .fit
    Use the fit method to create a Decision Tree Classifier on a given dataset.
    Data must be in the format of a list of lists,
    and include target values as the far value of each row of the dataset.

    .predict
    Use the predict method to generate predictions after fitting a tree.

    .accuracy
    Use the accuracy method to calculate the accuracy of your predictions.
    This method automatically calls the predict method, and can be used
    independently after a tree is fit.

    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def gini_impurity(self, groups, classes):
        """
        Calculates the Gini impurity for a given potential data split
        """
        # find number of observations at a given point
        n_instances = float(sum([len(group) for group in groups]))

        # find weighted Gini impurity for each
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # eliminate possibility of divide by zero error
            if size == 0:
                continue
            score = 0.0
            # find score for group based on score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group's score based on size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def find_split(self, index, value, dataset):
        """
        Sort data at a given split point based on proposed split value.
        """
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def evaluate_split(self, dataframe):
        """
        Evaluate potential splits to return the split with the best resulting
        improvements in Gini impurity score.
        """
        class_values = list(set(row[-1] for row in dataframe))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataframe[0])-1):
            for row in dataframe:
                groups = self.find_split(index, row[index], dataframe)
                gini = self.gini_impurity(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_leaf(self, group):
        """
        Creates a leaf node.
        """
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        """
        Generate data splits based on given node.
        """
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_leaf(left + right)
            return
        # check if beyond max depth
        if self.max_depth:
            if depth >= self.max_depth:
                node['left'], node['right'] = self.to_leaf(left), self.to_leaf(right)
                return

        # process left child
        if len(left) < self.min_samples_split:
            node['left'] = self.to_leaf(left)
        else:
            node['left'] = self.evaluate_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) < self.min_samples_split:
            node['right'] = self.to_leaf(right)
        else:
            node['right'] = self.evaluate_split(right)
            self.split(node['right'], depth+1)

    def fit(self, train):
        """
        Builds a decision tree based on input of training data.
        """
        root = self.evaluate_split(train)
        self.split(root, 1)
        return root

    def predict(self, node, row):
        """
        Makes a prediction for new data from a trained tree.
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def accuracy(self, node, dataset):
        """
        Generates an accuracy score by comparing expected versus actual
        predictions of classifications.
        """
        expected = []
        predictions = []
        for row in dataset:
            expect = row[-1]
            expected.append(expect)
            prediction = self.predict(node, row)
            predictions.append(prediction)

        # Find the number of correct predictions
        num_correct = sum(1 for x, y in zip(expected, predictions) if x == y)
        # Calculate accuracy
        accuracy = num_correct/len(expected)
        return accuracy
