#Install the package from pip install google-api-python-client
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import os

#service account based authentication, change the directory of the json

#Please update JSON file full path for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

#Input to be in a 2dimensional array,
instances=['768805383',45,'M','3','High School','Married','$60K - $80K','Blue',39,5,1,3,12691,777,11914,1.335,1144,42,1.625,0.061]

#Model details
project_name="ai-platform-01"
model_name="ChurnPredictor"
version="Version_1"

#Below lines of code is needed when we have deployed the model in the regional end point.
#Not required when model is deployed at global end point.

'''
#Replace region name accordingly.
endpoint = 'https://REGION-ml.googleapis.com' 
client_options = ClientOptions(api_endpoint=endpoint)
service = googleapiclient.discovery.build('ml', 'v1', client_options=client_options)
'''

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format(project_name,model_name,version)
response = service.projects().predict(
    name=name,
    body={'instances': instances}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
    print(response["predictions"])