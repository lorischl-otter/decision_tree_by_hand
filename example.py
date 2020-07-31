# Imports for manual example
import csv
from random import randrange
from tree import DecisionTreeClassifier

# Imports for pandas and sklearn comparison
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc

# =============================================================================
# Example of manual implementation
# =============================================================================

# Read in iris dataset
file = "iris.csv"
with open(file, 'r') as read_obj:
    csv_reader = csv.reader(read_obj, quoting=csv.QUOTE_NONE)
    list_of_rows = list(csv_reader)


# Manually split dataset into a train and test set
def train_test(dataset, split=0.75):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Split out training and test sets to use in model
train, test = train_test(list_of_rows[1:])

# Instantiate manual classifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=4)

# Fit / Create the decision tree
tree = clf.fit(train)

# Example of prediction generation
predictions = []
for row in list_of_rows[1:]:
    prediction = clf.predict(tree, row)
    predictions.append(prediction)

# Find accuracy of decision tree train & test data
training_accuracy = clf.accuracy(tree, train)
test_accuracy = clf.accuracy(tree, test)

print(f"Manual Training Accuracy: {training_accuracy:.2%}")
print(f"Manual Test Accuracy: {test_accuracy:.2%}")

# =============================================================================
# Compare to actual function using pandas and sklearn
# =============================================================================

df = pd.read_csv("iris.csv")
train, test = train_test_split(
    df, train_size=.75, stratify=df["species"], random_state=7)

target = ["species"]

X_train = train.drop(target, axis=1)
y_train = train[target]

X_test = test.drop(target, axis=1)
y_test = test[target]

clf = dtc(max_depth=5, min_samples_split=4)
clf.fit(X_train, y_train)

# Find accuracy of sklearn implementation
training_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f"Sklearn Train Score: {training_accuracy:.2%}")
print(f"Sklearn Test Score:, {test_accuracy:.2%}")
