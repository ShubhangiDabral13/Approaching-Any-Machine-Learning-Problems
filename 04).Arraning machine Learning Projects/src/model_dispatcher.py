# src/model_dispatcher.py

"""
model_dispatcher import various models from scikit-learn and defines dictionary with
keys that are anmes of the models and values are the models themselves.
"""
from sklearn import tree
from sklearn import ensemble

models = {
            "decision_tree_gini": tree.DecisionTreeClassifier( criterion="gini" ),
            "decision_tree_entropy": tree.DecisionTreeClassifier( criterion="entropy" ),
            "rf": ensemble.RandomForestClassifier(),
            }
