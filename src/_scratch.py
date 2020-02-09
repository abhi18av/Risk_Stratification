import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
# %matplotlib inline

os.chdir(
    "/Users/eklavya/projects/education/formalEducation/DataScience/DataScienceAssignments/HealthCare/Risk_Stratification/")


# Until fuction: line seperator
def print_ln():
    print('-' * 80, '\n')


# pd.options.display.float_format = '{:.2f}'.format


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

# DONE
"""
Remove redundant variables
"""

# Inspecting the head of the dataset
diabetic_patient_data.head(5)

# NOTE replace the `?` as `nan`
# https://stackoverflow.com/questions/52643775/how-to-replace-specific-character-in-pandas-column-with-null

diabetic_patient_data = diabetic_patient_data.replace('?', np.nan)
# diabetic_patient_data.to_csv("../_resources/diabetic_patient_data.csv", sep=',')


# DONE
"""
Check for missing values and treat them accordingly.
"""

# Analyse the missing values
columns_with_missing_data = round(100 * (diabetic_patient_data.isnull().sum() / len(diabetic_patient_data.index)), 2)
columns_with_missing_data[columns_with_missing_data > 20].plot(kind='bar')
plt.show()

# Three columns have considerable data missing
# - weight
# - payer_code
# - medical_speciality # TODO decide what to do

# We can see that Weight column is almost completely empty and therefore can be dropped
diabetic_patient_data = diabetic_patient_data.drop(['weight'], axis=1)

# `payer_code` is redundant for our purpose, so we can drop that as well
diabetic_patient_data = diabetic_patient_data.drop(['payer_code'], axis=1)

# DONE
"""
Change the variable 'readmitted' to binary type by clubbing the values ">30" and "<30" as "YES".
"""

diabetic_patient_data['readmitted'] = diabetic_patient_data['readmitted'].replace('>30', 'YES')
diabetic_patient_data['readmitted'] = diabetic_patient_data['readmitted'].replace('<30', 'YES')

# diabetic_patient_data.to_csv("../_resources/diabetic_patient_data.csv", sep=',')

# DONE
"""
Remove duplicated rows/columns
"""

# NOTE seems like there is no duplicated data

# deduplicated_patient_data = diabetic_patient_data.drop_duplicates()
# duplicated_data = diabetic_patient_data[diabetic_patient_data.duplicated()]

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
    # 'encounter_id',  # TODO can drop
    # 'patient_nbr',  # TODO can drop
    # 'admission_type_id',  # NOTE cat-encoded
    # 'discharge_disposition_id',  # NOTE cat-encoded
    # 'admission_source_id',  # NOTE cat-encoded
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

# NOTE Univariate analysis of some numerical attributes

for a_num_feature in diabetic_patient_data_num_features:
    sns.FacetGrid(diabetic_patient_data, hue='readmitted', size=6).map(sns.distplot, a_num_feature).add_legend()
    plt.show()

for a_num_feature in diabetic_patient_data_num_features:
    sns.BarPlot(diabetic_patient_data, hue='readmitted', size=6).map(sns.distplot, a_num_feature).add_legend()
    plt.show()

# Pairplot

diabetic_patient_data_num_features = [
    # 'encounter_id',  # TODO can drop
    # 'patient_nbr',  # TODO can drop
    # 'admission_type_id',  # NOTE cat-encoded
    # 'discharge_disposition_id',  # NOTE cat-encoded
    # 'admission_source_id',  # NOTE cat-encoded
    'readmitted',  # NOTE  add this for using with hue
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses']

diabetic_patient_data_num_features_df = diabetic_patient_data[diabetic_patient_data_num_features]
sns.pairplot(diabetic_patient_data_num_features_df, hue='readmitted').add_legend()
plt.show()

# TODO
"""
Perform basic data exploration for some categorical attributes
"""

diabetic_patient_data_cat_features = ['race',
                                      'gender',
                                      'age',  # TODO find out how to deal with these age ranges
                                      'medical_specialty',  # TODO  find out how to deal with this
                                      'diag_1',  # TODO find out how to deal with this
                                      'diag_2',  # TODO find out how to deal with this
                                      'diag_3',  # TODO find out how to deal with this
                                      'max_glu_serum',  # NOTE has low variance
                                      'A1Cresult',
                                      # diabetes-med-start # TODO these could be dropped or encoded in 0 or 1
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
                                      'readmitted']

# for a_med in [
#     'metformin',
#     'repaglinide',
#     'nateglinide',
#     'chlorpropamide',
#     'glimepiride',
#     'acetohexamide',
#     'glipizide',
#     'glyburide',
#     'tolbutamide',
#     'pioglitazone',
#     'rosiglitazone',
#     'acarbose',
#     'miglitol',
#     'troglitazone',
#     'tolazamide',
#     'examide',
#     'citoglipton',
#     'insulin',
#     'glyburide-metformin',
#     'glipizide-metformin',
#     'glimepiride-pioglitazone',
#     'metformin-rosiglitazone',
#     'metformin-pioglitazone']:
#     print(diabetic_patient_data[a_med].value_counts())

# ==================
# Data preparation
# ==================


# TODO
"""
Create dummy variables for categorical ones.
"""

# Create dummy variables for all the six categorial variables
night_pck_user_6_s = pd.get_dummies(telecom['night_pck_user_6'], prefix="night6", drop_first=True)
night_pck_user_7_s = pd.get_dummies(telecom['night_pck_user_7'], prefix="night7", drop_first=True)
night_pck_user_8_s = pd.get_dummies(telecom['night_pck_user_8'], prefix="night8", drop_first=True)

fb_user_6_s = pd.get_dummies(telecom['fb_user_6'], prefix="fb6", drop_first=True)
fb_user_7_s = pd.get_dummies(telecom['fb_user_7'], prefix="fb7", drop_first=True)
fb_user_8_s = pd.get_dummies(telecom['fb_user_8'], prefix="fb8", drop_first=True)

telecom = pd.concat([telecom, night_pck_user_6_s], axis=1)
telecom = pd.concat([telecom, night_pck_user_7_s], axis=1)
telecom = pd.concat([telecom, night_pck_user_8_s], axis=1)

telecom = pd.concat([telecom, fb_user_6_s], axis=1)
telecom = pd.concat([telecom, fb_user_7_s], axis=1)
telecom = pd.concat([telecom, fb_user_8_s], axis=1)

# drop all categorial columns
telecom = telecom.drop(
    ['fb_user_6', 'fb_user_7', 'fb_user_8', 'night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8'], 1)

# TODO
"""
Scale numeric attributes 
"""

# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns

# ==================
# Model Building
# ==================


# TODO
"""
Divide your data into training and testing dataset
"""

# split into train and test
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    test_size=0.3, random_state=100)

# TODO
"""
Train and compare the performance of at least two machine learning algorithms and decide which one to use for predicting risk of readmission for the patient.
Show important feature for each model is calculated.
"""

# This seems like a classification problem. And I'll rely on two models
# - Decision tree
# - Logistic regression


# TODO
"""
Use trained model to stratify your population into 3 risk buckets:

- High risk (Probability of readmission >0.7)
- Medium risk (0.3 < Probability of readmission < 0.7)
- Low risk (Probability of readmission < 0.3)
"""
