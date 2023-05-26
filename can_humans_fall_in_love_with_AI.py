import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
dataset = pd.read_csv(FILEPATH)
print(dataset)
dataset["gender_label"] = dataset.gender.astype("category").cat.codes
dataset["turing_label"] = dataset.c_turing_test.astype("category").cat.codes
dataset["pronouns_label"] = dataset.pronouns.astype("category").cat.codes
from sklearn.ensemble import RandomForestRegressor
features = ["chatgpt","characterAI","replika","aidungeon","novelai","kobold","laws_of_robotics","paperclip_maximizer","machine_learning","poi","turing_test","agi","please","friendship","gender_label","turing_label", "pronouns_label"]
target = "love"
print(dataset[target])
tree_classifier = DecisionTreeRegressor(max_depth=len(features))
tree_classifier.fit(dataset[features].values, dataset[target].values)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(100, 80))
plot_tree(tree_classifier,
          feature_names=features,
          rounded=True,
          filled=True,
          proportion=False)
