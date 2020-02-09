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
Remove redundant variables, duplicate rows/columns
"""

# TODO
"""
Check for missing values and treat them accordingly.

"""

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
