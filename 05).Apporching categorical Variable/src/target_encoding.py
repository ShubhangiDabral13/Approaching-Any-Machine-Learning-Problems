#target_encoding.py
import copy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def mean_target_encoding(data):

    #make a copy of dataframe
    df = copy.deepcopy(data)

    #list of numerical columns
    num_cols = [ "fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week" ]

    #map targets to 0s and 1s
    target_mapping = { " <=50K": 0, " >50K": 1 }

    df.loc[:, "income"] = df.income.map(target_mapping)

    

    #all columns are features except kfold & income columns
    features = [ f for f in df.columns if f not in ("kfold", "income")]

    #fill all NAN values with NONE
    #note that I am converting all columns to "strings"
    #it doesnt matter because all are categories
    for col in features:

        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    #now its time to label encode the features
    for col in features:

        if col not in num_cols:

            #initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()

            #fit label encoder on all data
            lbl.fit(df[col])

            #transform all the data
            df.loc[:, col] = lbl.transform(df[col])

    #a list to store 5 validation dataframes
    encoded_dfs = []

    #go over all folds
    for fold in range(5):

        #fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        #for all feature columns, i.e. categorical columns
        for column in features:
            #create dict of category:mean target
            mapping_dict = dict( df_train.groupby(column)["income"].mean() )

            #column_enc is the new column we have with mean encoding
            df_valid.loc[ :, column + "_enc" ] = df_valid[column].map(mapping_dict)

        #append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)

    #create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df

def run(df, fold):

    #note that folds are same as before
    #get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #all columns are features except income and kfold columns
    features = [ f for f in df.columns if f not in ("kfold", "income") ]

    #scale training data
    x_train = df_train[features].values

    #scale validation data
    x_valid = df_valid[features].values

    #initialize xgboost model
    model = xgb.XGBClassifier( n_jobs=-1, max_depth=7 )

    #fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)

    #predict on validation data #we need the probability values as we are calculating AUC
    #we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    #get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    #print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":

    df = pd.read_csv("adult_folds.csv")

    df = mean_target_encoding(df)

    for fold in range(5):
        run(df,fold)
