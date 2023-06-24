#!/usr/bin/env python
# coding: utf-8

# # Q1. What are missing values in a dataset? Why is it essential to handle missing values? Name some algorithms that are not affected by missing values
Missing values refer toi the absence of data in one or m ore columns or features in a dataset.
These missing values may occcur due to various reasons,such as data collection errors,data corruption.
#Some of the common algorithms that are not affected by missing values are:
# 1)Decision trees
# 2) Random Forests
# 3)Gradient Boosting Machines
# 4)K-Nearest neighbours
# 

# # Q2. List down techniques used to handle missing data. Give an example of each with python code

# #1)Deletion
: Deletion involves removing the rows or columns that contain missing values from the dataset. This technique is only recommended when the amount of missing data is small relative to the size of the dataset, and the missing data is missing completely at random (MCAR).
# In[5]:


import pandas as pd
import numpy as np
df=pd.DataFrame({'A': [1,2,np.nan,68],
                 'B': [5,np.nan,np.nan,2],
                 'C': [9,10,2,45]})
df.dropna(inplace=True)
df

Imputation: Imputation involves replacing the missing values with estimated values based on the available data. There are several methods of imputation, including mean imputation, median imputation, and mode imputation.
# In[6]:


import pandas as pd 
import numpy as np
df=pd.DataFrame({'A': [1,2,np.nan,68],
                 'B': [5,np.nan,np.nan,2],
                 'C': [9,10,2,45]})
df.fillna(df.mean(),inplace=True)
df

3)K-Nearest Neighbor Imputation: K-Nearest Neighbor imputation involves replacing missing values with the average of the K-nearest neighbors in the feature space.
# In[8]:


import pandas as pd
from sklearn.impute import KNNImputer

df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                   'B': [5, np.nan, np.nan, 8],
                   'C': [9, 10, 11, 12]})

imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df_imputed


# # 3. Explain the imbalanced data. What will happen if imbalanced data is not handled?
# 
# Imbalanced data refers to a dataset where the number of instances in one class is significantly higher or lower than the number of instances in another class. In other words, the distribution of the target variable is not uniform.
# 
# If imbalanced data is not handled, it can lead to several problems, including:
# 
# Biased model performance: In the case of imbalanced data, a model may be biased towards the majority class because it has more data to learn from. This can result in poor performance on the minority class, which may be of greater importance in certain applications.
# False positives and false negatives: In imbalanced data, a model may predict the majority class with high accuracy, but perform poorly on the minority class. This can lead to a high number of false positives and false negatives.
# Overfitting: In the case of imbalanced data, a model may overfit to the majority class, resulting in poor generalization performance.
Q4. What are Up-sampling and Down-sampling? Explain with an example when up-sampling and down-sampling are required.

Up-sampling refers to the process of increasing the number of instances in the minority class by randomly duplicating them. This can be done using techniques such as random oversampling or SMOTE (Synthetic Minority Over-sampling Technique).For example, suppose we have a dataset with 100 instances, out of which 90 belong to class A and 10 belong to class B. Since the dataset is imbalanced, we can up-sample the minority class B by randomly duplicating its instances, resulting in a new dataset with 100 instances, out of which 90 belong to class A and 20 belong to class B.

Down-sampling refers to the process of decreasing the number of instances in the majority class by randomly removing them. This can be done using techniques such as random undersampling or Tomek links. For example, suppose we have a dataset with 100 instances, out of which 90 belong to class A and 10 belong to class B. Since the dataset is imbalanced, we can down-sample the majority class A by randomly removing some of its instances, resulting in a new dataset with 50 instances, out of which 45 belong to class A and 5 belong to class B.
# Q5. What is data Augmentation? Explain SMOTE.
# 
# Data augmentation is a technique used to increase the size of a dataset by creating new, synthetic data from the original data. This is often done to address problems of overfitting, improve the generalization performance of machine learning models, or to balance an imbalanced dataset.
# 
# SMOTE (Synthetic Minority Over-sampling Technique) is a popular data augmentation technique. SMOTE works by creating synthetic instances of the minority class by interpolating between existing instances of that class. Specifically, for each instance in the minority class, SMOTE selects k nearest neighbors (typically k=5) and creates a new instance by linearly interpolating between the selected instance and one of its k nearest neighbors. The interpolation factor is chosen randomly between 0 and 1, and the new instance is added to the dataset. SMOTE is an effective technique for handling imbalanced data, as it creates new synthetic instances that are similar to existing instances in the minority class, and can thus help the machine learning model generalize better to the minority class.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




Q6. What are outliers in a dataset? Why is it essential to handle outliers?

Outliers are data points in a dataset that significantly deviate from the rest of the data points. They can be caused by errors in data collection, measurement errors, or they may represent actual extreme values in the population.

It is essential to handle outliers because they can have a significant impact on the performance of machine learning models. Outliers can affect the accuracy and generalization performance of a model, as they can bias the model towards the extreme values, leading to overfitting or underfitting. Outliers can also affect the results of statistical analysis, such as the mean, variance, and standard deviation, leading to inaccurate or misleading conclusions.

Q7. You are working on a project that requires analyzing customer data. However, you notice that some of the data is missing. What are some techniques you can use to handle the missing data in your analysis?

There are several techniques that can be used to handle missing data in customer data analysis:

Deletion: One technique is to simply delete the rows or columns with missing data. This is only recommended if the missing data is a small percentage of the total dataset and if the missing data is completely at random. However, if the missing data is a large percentage of the dataset, this technique can result in a significant loss of information.

Imputation: Another technique is to impute the missing values. This can be done using statistical methods such as mean imputation, median imputation, or mode imputation. Alternatively, more advanced techniques such as k-nearest neighbor imputation or multiple imputation can be used.

Machine learning-based methods: Machine learning-based methods can also be used to handle missing data. For example, regression-based methods can be used to predict missing values based on other variables in the dataset.

Domain knowledge: Domain knowledge can also be used to handle missing data. For example, if certain values are missing for a customer, but it is known that all customers with certain characteristics have the same value, then that value can be imputed for the missing values.
# Q8. You are working with a large dataset and find that a small percentage of the data is missing. What are some strategies you can use to determine if the missing data is missing at random or if there is a pattern to the missing data?
# 
# There are several strategies that can be used to determine if the missing data is missing at random or if there is a pattern to the missing data:
# 
# Visualization: One strategy is to visualize the data using plots and graphs to see if there are any patterns or trends in the missing data. For example, a heatmap can be used to show which values are missing in the dataset.
# 
# Summary statistics: Another strategy is to calculate summary statistics for the missing data and compare them to the summary statistics for the non-missing data. If there are significant differences between the two, this may suggest that the missing data is not missing at random.
# 
# Imputation: Imputation can also be used to determine if the missing data is missing at random. If the imputed values are similar to the non-missing values, this may suggest that the missing data is missing at random. If the imputed values are significantly different, this may suggest that there is a pattern to the missing data.

# # Q9. Suppose you are working on a medical diagnosis project and find that the majority of patients in the dataset do not have the condition of interest, while a small percentage do. What are some strategies you can use to evaluate the performance of your machine learning model on this imbalanced dataset?
# 
# Confusion Matrix: A confusion matrix is a table that summarizes the performance of a classifier on a dataset. It displays the number of true positives, false positives, true negatives, and false negatives. A confusion matrix can help to evaluate the performance of the model, especially when dealing with imbalanced datasets.
# 
# Precision, Recall, and F1-score: Precision, recall, and F1-score are metrics that are commonly used to evaluate the performance of machine learning models on imbalanced datasets. Precision is the fraction of true positive predictions among all positive predictions, while recall is the fraction of true positive predictions among all actual positive instances. F1-score is the harmonic mean of precision and recall. These metrics are useful when evaluating the performance of models on imbalanced datasets because they take into account both the false positive and false negative rates.
# 
# ROC Curve and AUC: ROC (Receiver Operating Characteristic) curve is a plot of the true positive rate against the false positive rate. AUC (Area Under the ROC Curve) is a metric that measures the area under the ROC curve. ROC curve and AUC can be used to evaluate the performance of a model on an imbalanced dataset. A high AUC value suggests that the model is performing well on the dataset.
# 
# Resampling techniques: Resampling techniques such as oversampling the minority class or undersampling the majority class can also be used to balance the dataset. Once the dataset is balanced, standard metrics such as accuracy, precision, recall, F1-score, and ROC can be used to evaluate the performance of the model.
Q10. When attempting to estimate customer satisfaction for a project, you discover that the dataset is unbalanced, with the bulk of customers reporting being satisfied. What methods can you employ to balance the dataset and down-sample the majority class?

Random undersampling: This method involves randomly selecting a subset of observations from the majority class to match the size of the minority class. This can be done using techniques such as RandomUnderSampler from the imblearn library in Python.

Tomek links: This method involves identifying pairs of observations that are nearest neighbors and belong to different classes. The observation from the majority class is then removed to balance the dataset. This can be done using techniques such as TomekLinks from the imblearn library in Python.

Synthetic minority oversampling technique (SMOTE): This method involves generating synthetic observations for the minority class to match the size of the majority class. SMOTE can be used in combination with random undersampling to balance the dataset. This can be done using techniques such as SMOTETomek from the imblearn library in Python.Q11. You discover that the dataset is unbalanced with a low percentage of occurrences while working on a project that requires you to estimate the occurrence of a rare event. What methods can you employ to balance the dataset and up-sample the minority class?

Random oversampling: This method involves randomly duplicating observations from the minority class to match the size of the majority class. This can be done using techniques such as RandomOverSampler from the imblearn library in Python.

Synthetic minority oversampling technique (SMOTE): This method involves generating synthetic observations for the minority class to match the size of the majority class. SMOTE can be used in combination with random oversampling to balance the dataset. This can be done using techniques such as SMOTE from the imblearn library in Python.

Adaptive synthetic sampling (ADASYN): This method is similar to SMOTE but focuses on generating more synthetic observations for the minority class samples that are harder to learn. This can be done using techniques such as ADASYN from the imblearn library in Python.
# In[ ]:




