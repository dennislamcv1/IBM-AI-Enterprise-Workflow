#!/usr/bin/env python

"""
Watson machine learning tutorial

NOTE: this script requires scikit-learn==0.20.3

"""


import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

from watson_machine_learning_client import WatsonMachineLearningAPIClient
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

## check for correct version of sklearn
SKLEARN_VERSION = "0.20.3"

from sklearn import __version__
if __version__ != SKLEARN_VERSION:
    raise Exception("sklearn version must be {}".format(SKLEARN_VERSION))

def load_wml_credentials():
    """
    fetch the saved watson machine learning credentials
    """

    wmlcreds_dir = os.path.join(os.path.expanduser("~"),".ibm")
    wmlcreds_file = os.path.join(wmlcreds_dir,'wml.json')

    if not os.path.exists(wmlcreds_file):
        raise Exception("cannot find {}".format(wmlcreds_file))
    
    with open(wmlcreds_file, "r") as read_file:
        wmlcreds = json.load(read_file)

    return(wmlcreds)
        
def connect_wml_service():
    """
    Instantiate a client using credentials
    """

    wmlcreds = load_wml_credentials()
    wml_credentials = {
        "apikey": wmlcreds['apikey'],
        "instance_id": wmlcreds['instance_id'],
        "url": wmlcreds['url'],
    }

    client = WatsonMachineLearningAPIClient(wml_credentials)
    return(client)

def get_preprocessor():
    """
    create a preprocessor for the pipeline
    """
    
    ## preprocessing pipeline
    numeric_features = ['age', 'num_streams']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])

    categorical_features = ['country', 'subscriber_type']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])
    return(preprocessor)


def load_aavail_data():
    data_dir = os.path.join(".","data")
    df = pd.read_csv(os.path.join(data_dir,r"aavail-target.csv"))
       
    ## pull out the target and remove uneeded columns
    _y = df.pop('is_subscriber')
    y = np.zeros(_y.size)
    y[_y==0] = 1 
    df.drop(columns=['customer_id','customer_name'],inplace=True)
    df.head()
    X = df

    return(X,y)

if __name__ == "__main__":


    ## connect to the wml service with saved credentials
    client = connect_wml_service()

    ## prepare the data
    X,y = load_aavail_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    ## prepare the model
    params = {'n_estimators': 100,'max_depth':2}   
    clf = ensemble.RandomForestClassifier(**params)
    preprocessor = get_preprocessor()
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])

    ## train the model
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test,y_pred))

    ## save the model in the WML service
    metadata = {
    client.repository.ModelMetaNames.NAME: "wml-tutorial-aavail",
    client.repository.ModelMetaNames.FRAMEWORK_NAME: "scikit-learn",
    client.repository.ModelMetaNames.FRAMEWORK_VERSION: SKLEARN_VERSION
    }
    model_details = client.repository.store_model(pipe, meta_props=metadata)
    print("model saved successfully")


    
    ## deploy model in th WML service
    model_uid = client.repository.get_model_uid(model_details)
    model_deployment_details = client.deployments.create( artifact_uid=model_uid, name="test-deploy" )
    print("model deployed successfully")
    

