# A Simple Decision Tree Classifier Implementation
Manual Python implementation of a decision tree classifier

See `tree.py` for source code, and `example.py` for sample implementation with the iris dataset. 

This simple decision tree classifier class is only built for numeric or ranked data. It accepts a list of lists as its data input, with classifications (string or numeric) included as the last element of each list or row. It was built without, and can be implemented without, pandas or sklearn. It is hard-coded to use gini impurity as its split metric/criterion.

The performance of this decision tree on the iris dataset is comparable to sklearn's implementation, both scoring around 95% test accuracy with a `max_depth` of 5 and a `min_samples_split` of 4.

For additional information about the implementation, see my [Medium blog post.](https://medium.com/@lori.schlatter/implementing-a-decision-tree-classifier-from-scratch-in-python-6475adf4470c?sk=126da4d289d68f4f45597a499e39bbaf)

The code for this decision tree was adapted and expanded as an exercise, from [Jason Brownlee's Machine Learning Mastery.](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)