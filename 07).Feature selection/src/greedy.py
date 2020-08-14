#greedy.py
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:

    """
    A simple and custom class for greedy feature selection.
    You will need to modify it quite a bit to make it suitable
    for your dataset.
    """

    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns
        Area Under ROC Curve (AUC) NOTE: We fit the data and
        calculate AUC on same data. WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection.
        k-fold will take k times longer.

        If you want to implement it in really correct way,
        calculate OOF AUC and return mean AUC over k folds.
        This requires only a few lines of change and has been
        shown a few times in this book.

        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """

        #fit the logistic regression model,
        #and calculate AUC on same data
        #again: BEWARE
        #you can choose any model that suits your data

        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, X, y):

        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """

        #initialize good features list
        #and best scores to keep track of both
        good_features = []
        best_scores = []

        #calculate the number of features
        num_features = X.shape[1]

        #infinite loop
        while True:
            #initialize best feature and score of this loop
            this_feature = None
            best_score = 0

            #loop over all features
            for feature in range(num_features):
                #if feature is already in good features,
                #skip this for loop
                if feature in good_features:
                    continue

                #selected features are all good features till now
                #and current feature
                selected_features = good_features + [feature]
                #remove all other features from data
                xtrain = X[:, selected_features]

                #calculate the score, in our case, AUC

                score = self.evaluate_score(xtrain, y)

                #if score is greater than the best score
                #of this loop, change best score and best feature
                if score > best_score:
                    this_feature = feature
                    best_score = score

            #if we have selected a feature, add it
            #to the good feature list and update best scores list
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            #if we didnt improve during the last two rounds,
            #exit the while loop
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break

        #return best scores and good features
        #why do we remove the last data point?
        return best_scores[:-1], good_features[:-1]

    def _call(self, X, y):

        """
        Call function will call the class on a set of arguments
        """

        #select features, return scores and selected indices scores,
        features,score = self._feature_selection(X, y)
        #transform data with selected features
        return features, score

if __name__ == "__main__":

    #generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    #transform data by greedy feature selection
    obj = GreedyFeatureSelection()
    features,score = obj._call(X,y)
    print("The number of features are {}".format(len(features)))
    print()
    print("The features are {}".format(features))
    print()
    print("The AUC score are {}".format(score))
