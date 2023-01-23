#Python package for google AutoML
#google-cloud-automl
from google.cloud import automl
import os

#For google authentication purpose.
#Update full path of service account credential file (JSON file)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

project_id = 'machine-learning-322713'
model_id = "ICN1344172756165459968"
file_path = r"C:\Users\HEMANTH KUMAR K\Desktop\IMAGE TESTING\028.jpg"
compute_region = 'us-central1'

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(project_id, compute_region, model_id)
print(model_full_id)
# Read the file.
with open(file_path, "rb") as content_file:
    content = content_file.read()


image = automl.Image(image_bytes=content)
payload = automl.ExamplePayload(image=image)

params = {"score_threshold": "0.5"}

request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
response = prediction_client.predict(request=request)

print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))