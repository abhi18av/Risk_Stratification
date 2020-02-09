import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %matplotlib inline

# Until fuction: line seperator
def print_ln():
    print('-' * 80, '\n')


# pd.options.display.float_format = '{:.2f}'.format

# ==================
# Data preparation
# ==================

# Loading in the dataset
diabetic_patient_data_orig = pd.read_csv('../resources/diabetic_data.csv')

# Create a working copy
diabetic_patient_data = diabetic_patient_data_orig.copy()

# Exploring the shape and info about the dataset
print('Dataframe Shape: ', diabetic_patient_data.shape)
print_ln()
print("Dataframe Info: \n")
diabetic_patient_data.info()
print_ln()

# Inspecting the head of the dataset
diabetic_patient_data.head(5)


# Identifying the numerical and categorical features
def type_features(data):
    categorical_features = data.select_dtypes(include=["object"]).columns
    numerical_features = data.select_dtypes(exclude=["object"]).columns
    print("categorical_features :", categorical_features)
    print_ln()
    print("numerical_features:", numerical_features)
    print_ln()
    return categorical_features, numerical_features


diabetic_patient_data_cat_features, diabetic_patient_data_num_features = type_features(diabetic_patient_data)

# TODO
"""
Remove redundant variables
"""
# NOTE replace the `?` as `nan`
# https://stackoverflow.com/questions/52643775/how-to-replace-specific-character-in-pandas-column-with-null

# diabetic_patient_data['weight'] = diabetic_patient_data['weight'].replace('?', np.nan)

diabetic_patient_data = diabetic_patient_data.replace('?', np.nan)

"""
Remove duplicated rows/columns
"""

duplicated_data = diabetic_patient_data[diabetic_patient_data.duplicated()]  # seems like there is no duplicated data

# TODO
"""
Check for missing values and treat them accordingly.

"""

# Analyse the missing values
diabetic_patient_data.isnull().sum()  # seems like there is no missing data

# TODO
"""
Scale numeric attributes and create dummy variables for categorical ones.

"""

# TODO
"""
Change the variable 'readmitted' to binary type by clubbing the values ">30" and "<30" as "YES".

"""

# TODO
"""
Create the derived metric 'comorbidity', according to the following scheme -

"""

# ==================
# Data Exploration
# ==================


# TODO
"""
Perform basic data exploration for some categorical attributes
"""

# TODO
"""
Perform basic data exploration for some numerical attributes
"""

# ==================
# Model Building
# ==================


# TODO
"""
Divide your data into training and testing dataset
"""
# TODO
"""
Train and compare the performance of at least two machine learning algorithms and decide which one to use for predicting risk of readmission for the patient.
"""
# TODO
"""
Important feature for each model is calculated.
"""
# TODO
"""
Use trained model to stratify your population into 3 risk buckets:
"""
# TODO
"""
High risk (Probability of readmission >0.7)
"""
# TODO
"""
Medium risk (0.3 < Probability of readmission < 0.7)
"""
# TODO
"""
Low risk (Probability of readmission < 0.3)
"""
