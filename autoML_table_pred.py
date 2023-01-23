from typing import Dict

#Install google-cloud-aiplatform

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os

def predict_tabular_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com"):

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )


    predictions = response.predictions
    print(predictions)
    for prediction in predictions:
        print(" prediction:", dict(prediction))

#Authentication using service account.
#Please update full path of JSON file for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =""

inputs = {'brand': 'Maruti','model':'Alto','min_cost_price':'357003.861','max_cost_price':'465401.5444','vehicle_age':'9','km_driven':'120000','seller_type':'Individual','fuel_type':'Petrol','transmission_type':'Manual','mileage':'19.7','engine':'796','max_power':'46.3','seats':'5'}
predict_tabular_sample("vertex-ai-123",919094963797491712,inputs)