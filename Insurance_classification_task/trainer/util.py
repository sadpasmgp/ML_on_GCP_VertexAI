from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage


#Names of all the columns including the target
COLUMNS_Name = ['id','Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage','Response']

#Target column for prediction
LABEL_COLUMN = 'Response'

COLUMNS_Drop = ['id']


#All the categorical data information is stored in the dictionaries along with the values of each columns for data preprocessing
#Instead of providing numerical values for each category, one-hot encoding can also be applied.
CATEGORICAL_COL = {
    'Gender': pd.api.types.CategoricalDtype(categories=['Male', 'Female']),
    'Vehicle_Age': pd.api.types.CategoricalDtype(categories=['< 1 Year', '> 2 Years', '1-2 Year']),
    'Vehicle_Damage': pd.api.types.CategoricalDtype([
        'Yes', 'No'])
}


def preprocess(dataframe):
    #Takes dataframe as the input, applies data transformation on numerical and categorical columns
    #Returns back a dataframe

    #To drop the id column from the data set
    dataframe = dataframe.drop(columns=COLUMNS_Drop)
    # Numerical data is converted to float32
    numeric_columns = dataframe.select_dtypes(['int64']).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

    # Convert categorical columns to numerictype
    cat_columns = dataframe.select_dtypes(['object']).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
        CATEGORICAL_COL[x.name]))
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
    return dataframe


def standardize(dataframe):
    #Takes dataframe as the input, applies z-score transformation on only the numerical data
    #Returns:Dataframe

    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes: #only for numerical columns,
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data():
    #Loads the data from the cloud storage and pre-preprocess the data and splits the data into training and testing datasets

    #Authentication before accessing the data stored on the cloud storage
    #Change the path of the json (service account) as per needs
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'E:\UDEMY\GCP\Service_account\gcp-ml-demo1-e29cfb0662dd.json'

    #Bucket name -> aiplatform_demo
    storage_client = storage.Client()
    public_bucket = storage_client.bucket('aiplatform_demo')
    blob = public_bucket.blob('Insurance_Train_data.csv')
    blob.download_to_filename('Insurance_Train_data.csv')
    train_df=pd.read_csv('./Insurance_Train_data.csv',sep=',')
    blob = public_bucket.blob('Insurance_Test_data.csv')
    blob.download_to_filename('Insurance_Test_data.csv')
    test_df=pd.read_csv('./Insurance_Test_data.csv',sep=',',)
    '''
    train_df = pd.read_csv(r'gs://aiplatform_demo/Jobs/data/Insurance_Train_data.csv')

    test_df = pd.read_csv(r'gs://aiplatform_demo/Jobs/data/Insurance_Test_data.csv')
    '''
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Split train and test data with labels. The pop method copies and removes
    # the label column from the dataframe.
    train_x, train_y = train_df, train_df.pop(LABEL_COLUMN)
    test_x, test_y = test_df, test_df.pop(LABEL_COLUMN)

    # Join train_x and test_x to normalize on overall means and standard
    # deviations. Then separate them again.
    #Can be also be done using fit transform on the train data and transform on the test data
    x1 = pd.concat([train_x, test_x], keys=['train', 'test'])
    x1= standardize(x1)
    train_x, test_x = x1.xs('train'), x1.xs('test')

    # Re-shaping the label column data into numpy arrays
    train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
    test_y = np.asarray(test_y).astype('float32').reshape((-1, 1))

    return train_x, train_y, test_x, test_y

