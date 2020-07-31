# A Simple Decision Tree Classifier Implementation
Manual Python implementation of a decision tree

See `tree.py` for source code, and `example.py` for sample implementation with the iris dataset. 

This simple decision tree classifier class is only built for numeric or ranked data, and it accepts a list of lists as its data input, with classifications (string or numeric) included as the last element of each list or row. It was built without, and can be implemented without, pandas or sklearn. 

The performance of this decision tree on the iris dataset is comparable to sklearn's implementation, both scoring around 95% test accuracy with a `max_depth` of 5 and a `min_samples_split` of 4.

The code for this decision tree was adapted and expanded from Jason Brownlee's Machine Learning Mastery article located here: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/