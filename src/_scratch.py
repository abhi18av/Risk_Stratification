import pandas as pd
import seaborn as sns
import numpy as np

# ==================
# Data preparation
# ==================

"""


Remove redundant variables, duplicate rows/columns

Check for missing values and treat them accordingly.

Scale numeric attributes and create dummy variables for categorical ones.

Change the variable 'readmitted' to binary type by clubbing the values ">30" and "<30" as "YES".

Create the derived metric 'comorbidity', according to the following scheme -


"""
print("Hello, Python")
