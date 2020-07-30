import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    #load the full training data with folds
    df = pd.read_csv("adult_folds.csv")

    #list of numerical columns
    num_cols = [
    "age",
    "fnlwgt",
    "capital.gain",
    "capital.loss",
    "hours.per.week"
    ]

    #drop numerical columns
    df = df.drop(num_cols,axis = 1)

    #map target to 0s and 1s
    target_mapping = {
    " <=50K": 0,
    " >50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)
    

    #all columns are features except income and kfold columns
    features = [
    f for f in df.columns if f not in ("kfold","income")
    ]

    #fill all NaN values with NONE
    #note that i am converting all columns to "string"
    # it doesnt matter beccause all are categories

    for col in features:
        df.loc[ : ,col] = df[col].astype(str).fillna("NONE")

    #get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop = True)

    #get training data using folds
    df_valid = df[df.kfold == fold].reset_index(drop = True)


    # initialze OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    #fit ohe on training + validation features
    full_data = pd.concat([df_train[features] ,df_valid[features]],axis = 0)

    ohe.fit(full_data[features])

    #transform training data
    x_train = ohe.transform(df_train[features])

    #transform validation data
    x_valid = ohe.transform(df_valid[features])

    #initialize Logistic Regression model
    model = linear_model.LogisticRegression()
    #fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)


    #predict on validation data
    #we need the probability values as we are calculating AUC
    #we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    #get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    #print auc
    print(f"Fold = {fold}, AUC = {auc}")



if __name__ == "__main__" :
    #run function for fold = 1
    for fold_ in range(5):
        run(fold_)
