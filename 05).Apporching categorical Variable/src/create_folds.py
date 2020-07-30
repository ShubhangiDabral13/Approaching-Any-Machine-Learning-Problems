#create_folds.py
#import pandas and model_selection module of scikit-learn

import pandas as  pd
from sklearn import model_selection

if __name__ == "__main__":

    # Read training data
    df = pd.read_csv("adults.csv")

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the necxt step id to randomize the rows of the data
    df = df.sample(frac= 1).reset_index(drop = True)

    # fetch labels
    y = df.income.values

    #initialze the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)

    #fill the new kfold column

    for f, (t_,v_) in enumerate(kf.split(X = df, y = y)):
        df.loc[v_,"kfold"] = f


    #save the new csv with kfold column
    df.to_csv("adult_folds.csv",index = False)
