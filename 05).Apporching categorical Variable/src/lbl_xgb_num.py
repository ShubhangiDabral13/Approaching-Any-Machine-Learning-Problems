import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    #load the full training full data
    df = pd.read_csv("adult_folds.csv")

    #list of numerical columns
    num_cols = [ "fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week" ]

    #map targets to 0s and 1s
    target_mapping = { " <=50K": 0, " >50K": 1 }

    df.loc[:, "income"] = df.income.map(target_mapping)

    #all columns are features except kfold & income columns
    features = [ f for f in df.columns if f not in ("kfold", "income") ]

    #fill all NaN values with NONE
    #note that I am converting all columns to "strings"
    #it doesnt matter because all are categories
    for col in features:
        #do not encode the numerical columns if col not in
        num_cols: df.loc[:, col] = df[col].astype(str).fillna("NONE")

        #now its time to label encode the features
        for col in features:
            if col not in num_cols:
                #initialize LabelEncoder for each feature column
                lbl = preprocessing.LabelEncoder()

                #fit label encoder on all data
                lbl.fit(df[col])

                #transform all the data
                df.loc[:, col] = lbl.transform(df[col])

    #get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #get training data
    x_train = df_train[features].values

    #get validation data
    x_valid = df_valid[features].values

    #initialize xgboost model
    model = xgb.XGBClassifier( n_jobs=-1 )

    #fit model on training data (ohe) \
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
