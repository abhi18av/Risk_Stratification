import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

os.chdir(
    "/Users/eklavya/projects/education/formalEducation/DataScience/DataScienceAssignments/HealthCare/Risk_Stratification/")


# Util fuction: line seperator
def print_ln():
    print('-' * 80, '\n')


# ==================
# Data Exploration
# ==================


# Loading in the dataset
diabetic_patient_data_orig = pd.read_csv('./resources/diabetic_data.csv')

# Create a working copy
diabetic_patient_data = diabetic_patient_data_orig.copy()

# Exploring the shape and info about the dataset
print('Dataframe Shape: ', diabetic_patient_data.shape)
print_ln()
print("Dataframe Info: \n")
diabetic_patient_data.info()
print_ln()

diabetic_patient_data = diabetic_patient_data.replace('?', np.nan)
diabetic_patient_data['medical_specialty'].replace({np.nan: 'Unknown'}, inplace=True)

diabetic_patient_data = diabetic_patient_data.drop(['encounter_id'], axis=1)
diabetic_patient_data = diabetic_patient_data.drop(['patient_nbr'], axis=1)
diabetic_patient_data = diabetic_patient_data.drop(['weight'], axis=1)
diabetic_patient_data = diabetic_patient_data.drop(['payer_code'], axis=1)

diabetic_patient_data = diabetic_patient_data.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)

diabetic_patient_data['readmitted'] = diabetic_patient_data['readmitted'].replace('>30', 'YES')
diabetic_patient_data['readmitted'] = diabetic_patient_data['readmitted'].replace('<30', 'YES')

diabetic_patient_data = diabetic_patient_data.drop_duplicates()


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
Perform basic data exploration for some numerical attributes
"""

diabetic_patient_data_num_features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses']

diabetic_patient_data_num_features_df = diabetic_patient_data[diabetic_patient_data_num_features]

diabetic_patient_data_num_features_df.describe()

diabetic_patient_data_num_features_df.info()

# Checking whether data is imbalanced or not
readmitted_df = diabetic_patient_data["readmitted"].value_counts()

diabetic_patient_data_rate = readmitted_df[1] / (readmitted_df[1] + readmitted_df[0])

# TODO
"""
Perform basic data exploration for some categorical attributes
"""

diabetic_patient_data_cat_features = ['race',
                                      'gender',
                                      'age',
                                      'medical_specialty',
                                      'max_glu_serum',
                                      'A1Cresult',
                                      # diabetes-med-start
                                      'metformin',
                                      'repaglinide',
                                      'nateglinide',
                                      'chlorpropamide',
                                      'glimepiride',
                                      'acetohexamide',
                                      'glipizide',
                                      'glyburide',
                                      'tolbutamide',
                                      'pioglitazone',
                                      'rosiglitazone',
                                      'acarbose',
                                      'miglitol',
                                      'troglitazone',
                                      'tolazamide',
                                      'examide',
                                      'citoglipton',
                                      'insulin',
                                      'glyburide-metformin',
                                      'glipizide-metformin',
                                      'glimepiride-pioglitazone',
                                      'metformin-rosiglitazone',
                                      'metformin-pioglitazone',
                                      # diabetes-med-end
                                      'change',
                                      'diabetesMed',
                                      'readmitted'
                                      ]


for a_cat_feat in diabetic_patient_data_cat_features:
    print(diabetic_patient_data[a_cat_feat].value_counts().count())
    print(diabetic_patient_data[a_cat_feat].value_counts())
    print_ln()

# ==================
# Data preparation
# ==================

# DONE
"""
Scale numeric attributes 
"""

# scaling the features
from sklearn.preprocessing import scale

for a_num_feat in diabetic_patient_data_num_features:
    diabetic_patient_data[a_num_feat] = pd.DataFrame(scale(diabetic_patient_data[a_num_feat]))

# DONE
"""
Create dummy for categorical attributes 
"""

for a_cat_feat in diabetic_patient_data_cat_features:
    tmp = pd.get_dummies(diabetic_patient_data[a_cat_feat], prefix=a_cat_feat, drop_first=True)
    diabetic_patient_data = pd.concat([diabetic_patient_data, tmp], axis=1)
    diabetic_patient_data = diabetic_patient_data.drop([a_cat_feat], 1)

diabetic_patient_data.info()


def final_cleanup(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


final_cleanup(diabetic_patient_data)

# DONE
"""
Split the cleansed dataset for the analysis
"""

from sklearn.model_selection import train_test_split

y = diabetic_patient_data.loc[:, 'readmitted_YES']
X = diabetic_patient_data.loc[:, diabetic_patient_data.columns != 'readmitted_YES']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    test_size=0.3, random_state=100)

# TODO
"""
Build a decision tree model
"""

# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)

# Let's check the evaluation metrics of our default model

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix

# Making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classification report
print(classification_report(y_test, y_pred_default))
