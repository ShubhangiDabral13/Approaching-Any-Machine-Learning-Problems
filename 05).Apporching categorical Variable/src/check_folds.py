import pandas as pd
df = pd.read_csv("train_folds.csv")

print("Number of samples in each folds\n")
print(df.kfold.value_counts())

print("Target in kfold = 0")
print(df[df.kfold == 0].target.value_counts())

print("Target in kfold = 1")
print(df[df.kfold == 1].target.value_counts())

print("Target in kfold = 2")
print(df[df.kfold == 2].target.value_counts())

print("Target in kfold = 3")
print(df[df.kfold == 3].target.value_counts())

print("Target in kfold = 4")
print(df[df.kfold == 4].target.value_counts())
