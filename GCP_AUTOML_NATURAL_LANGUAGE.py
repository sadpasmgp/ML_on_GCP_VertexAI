from google.cloud import automl
import os

#For google authentication purpose.
#Update full path of service account credential file (JSON file)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

project_id = 'machine-learning-322713'
model_id = "TCN2932676787232047104"
content = "This maid of mine who comes to wash dishes at home is usually late. " \
          "This morning I was in a hurry to go to the bank for some important work. " \
          "Well what do you know! The maid came right on time. The dishes were cleaned. " \
          "I did not want to come home to a mess."
compute_region = 'us-central1'

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1", model_id)

text_snippet = automl.TextSnippet(content=content, mime_type="text/plain")
payload = automl.ExamplePayload(text_snippet=text_snippet)

response = prediction_client.predict(name=model_full_id, payload=payload)

print(content)
for annotation_payload in response.payload:

    print(u"Predicted class name: {}".format(annotation_payload.display_name))
    print(
        u"Predicted class score: {}".format(annotation_payload.classification.score)
    )