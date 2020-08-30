#!/usr/bin/env python

"""
Example to showcase how to use the Watson NLU service

to store you API key

~$ mkdir ~/.ibm
~$ touch ~/.ibm/ibmauth.py

then edit the file to contain

ibmauth_key = "your api key"

"""



import sys
import os
import json
from ibm_watson import VisualRecognitionV3
from ibm_watson.visual_recognition_v4 import FileWithMetadata, TrainingDataObject, Location, AnalyzeEnums
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

### import API key
apikey_dir = os.path.join(os.path.expanduser("~"),".ibm")
sys.path.append(apikey_dir)

if not os.path.exists(apikey_dir):
    raise Exception("please store you API key in file within 'apikey_dir' before proceeding")

from ibmauth import VR_KEY, VR_URL, VR_VERSION

def connect_watson_vr():
    """
    establish a connection to watson vr service
    """
    
    authenticator = IAMAuthenticator(VR_KEY)
    service = VisualRecognitionV3(version=VR_VERSION,
                                  authenticator=authenticator)

    service.set_service_url(VR_URL)

    print("\nConnection established.\n")
    return(service)



if __name__ == "__main__":
    

    
    service = connect_watson_vr()

    ## classify an image from a URL
    image_url = "https://watson-developer-cloud.github.io/doc-tutorial-downloads/visual-recognition/fruitbowl.jpg"
    fruitbowl_results = service.classify(url=image_url,
                                         threshold='0.1',
                                         classifier_ids=['food']).get_result()
    print(json.dumps(fruitbowl_results, indent=2))
