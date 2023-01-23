#Install the package from pip install google-api-python-client
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import os

#service account based authentication, change the directory of the json


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'E:\SERVICE_ACCOUNT\ai-platform-01-573ffec6b0e4.json'

#Input to be in a 2dimensional array,
instances=[[6.1,3.2,5.7,2.5],[5.1,3.5,1.4,0.5],[6.7,3,5,1.7]]

#Model details
project_name="ai-platform-01"
model_name="ai_platform_iris_deployment"
version="Version1"

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
    for i in range(len(instances)):
        print("Output for input {} is {}".format(instances[i],response["predictions"][i]))
