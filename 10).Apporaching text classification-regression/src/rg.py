#import what we need
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing


if __name__ == "__main__":

    #read the training data
    df = pd.read_csv("IMDB.csv")

    #map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply( lambda x: 1 if x == "positive" else 0 )

    #we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    #the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    #fetch labels
    y = df.sentiment.values

    #initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    #fill the new kfold column
    for f, (t, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    #we go over the folds created
    for fold_ in range(5):
        #temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)


        #initialize CountVectorizer with NLTK's word_tokenize
        #function as tokenizer
        count_vec = CountVectorizer( tokenizer=word_tokenize, token_pattern=None)

        #fit count_vec on training data reviews
        count_vec.fit(train_df.review)

        #transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        #initialize logistic regression model

        model = linear_model.LogisticRegression(dual = True)
        #fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)

        #make predictions on test data
        #threshold for predictions is 0.5
        preds = model.predict(xtest)

        #calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
