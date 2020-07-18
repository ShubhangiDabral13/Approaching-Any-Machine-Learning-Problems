# src/creat_folds.py

import pandas as pd
import numpy as np
from sklearn import model_selection


def creat_fold(df):

    """
     param df: it is a dataframe which consist of the information of mnist_train.csv file from the input folder.
     creat_fold will create a new csv file mnist_train_folds.csv.
     The only difference between the mnist_train.csv and mnist_train_folds.csv is that mnist_train_folds.csv is shuffled and has a new column kfolds. 
    """

    print(df.head())

    #we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    #the next step is to randomize the rows of the data
    df = df.sample(frac = 1).reset_index(drop = True)

    #initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits = 5)

    #fill the new kfold column
    for fold, (trn_,val_) in enumerate(kf.split(X = df)):
        df.loc[val_,"kfold"]  = fold

        #Save the file
        df.to_csv("../input/mnist_train_folds.csv")



#reading the mnist_train.csv file from input folder.
df  = pd.read_csv("../input/mnist_train.csv")
creat_fold(df)
