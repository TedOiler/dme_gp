#--- load required packages
import os
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# #--- load data
# cuisines_path = os.path.join(os.getcwd(), 'recipes', 'Cuisines.csv')
# complete_path = os.path.join(os.getcwd(), 'recipes', 'recipes-mallet.txt')
# testing_path = os.path.join(os.getcwd(), 'recipes', 'Queries', 'recipes-mallet-testing.txt')
# training_path = os.path.join(os.getcwd(), 'recipes', 'Queries', 'recipes-mallet-training.txt')

# cuisines = pd.read_csv(cuisines_path, delimiter=',')
# #complete_data = pd.read_csv(complete_path, delimiter=',')
# #testing_data = pd.read_csv(testing_path, delimiter=',')
# training_data = pd.read_csv(training_path, delimiter=',', header=None)
# print(cuisines)

#--- load data
ing_mat = loadmat('recipes/MATLAB/ingredients.mat')['ingredients']
cityDist_mat = loadmat('recipes/MATLAB/citiesDistMat.mat')['citiesDistMat']
labelName_mat = loadmat('recipes/MATLAB/labelNames.mat')['labelNames']
labels_mat = loadmat('recipes/MATLAB/labels.mat')['labels']
recipe_mat = loadmat('recipes/MATLAB/recipes.mat')['recipes']

#--- for colnames
ing_headline = []
for i in ing_mat[0]:
    ing_headline.append(i[0])
#--- create data matrices
dataset_X = pd.DataFrame(recipe_mat,columns=ing_headline) #predictors
dataset_y = pd.DataFrame(labels_mat,columns=['label']) #labels
X_train_full,X_test,y_train_full,y_test = train_test_split(dataset_X,dataset_y,test_size=0.2) #train test split
X_train,X_val,y_train,y_val = train_test_split(X_train_full,y_train_full,test_size=0.25) #train val split
X_train_len = len(X_train)

freq_ing = dataset_X.sum()

print(head(dataset_X)) 

print(type(freq_ing))

print(sum(dataset_X["acorn squash"]))
print(dataset_X.sum())
