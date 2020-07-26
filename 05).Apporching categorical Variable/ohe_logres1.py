import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with train_folds
    df = pd.read_csv("train_folds.csv")

    #all columns are features except id, target and kfolds column

    features = [f for f in df.columns if f not in ("id" , "target" , "kfold")]


    #fill all the NaN with NONE
    #converting all columns to string

    for col in features:

        df.loc[:,col] = df[col].astype(str).fillna("NONE")


    #get training data using kfolds
    df_train = df[df.kfold  != fold].reset_index(drop = True)

    #get validation data using folds
    df_valid =  df[df.kfold  == fold].reset_index(drop = True)

    # initialze OneHotEncoding from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    #fit ohe on training + validation features
    full_data = pd.concat([df_train[features],df_valid[features]],axis = 0)

    ohe.fit(full_data[features])

    #Transform trainig data
    x_train = ohe.transform(df_train[features])

    #Transform validation data
    x_valid = ohe.transform(df_valid[features])


    #initialze logistic regression model
    model = LogisticRegression()

    #fit the model on training data (ohe)
    model.fit(x_train,df_train.target.values)

    #predict on validation data
    # we need the probabilty values as we are calculating AUC
    # we will use the probabilty of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]

    # get roc s=auc score
    auc = metrics.roc_auc_score(df_valid.target.values,valid_preds)

    #print AUC
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__" :
    #run function for fold = 1
    for fold_ in range(5):
        run(fold_)
