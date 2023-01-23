#Python package for google AutoML
#google-cloud-automl

from google.cloud import automl_v1beta1 as automl
import os
#Update full path of service account credential file (JSON file)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

project_id = 'machine-learning-322713'
compute_region = 'us-central1'
model_display_name = 'Price_pred_model1'
inputs = {'brand': 'Maruti','model':'Alto','min_cost_price':357003.861,'max_cost_price':465401.5444,'vehicle_age':9,'km_driven':120000,'seller_type':'Individual','fuel_type':'Petrol','transmission_type':'Manual','mileage':19.7,'engine':796,'max_power':46.3,'seats':5}
client = automl.TablesClient(project=project_id, region=compute_region)

feature_importance=False

if feature_importance:
    response = client.predict(
        model_display_name=model_display_name,
        inputs=inputs,
        feature_importance=feature_importance,
    )
else:
    response = client.predict(
        model_display_name=model_display_name, inputs=inputs
    )




for result in response.payload:
    print("Predicted price of the car: {}".format(result.tables.value))


    if feature_importance:
        # get features of top importance
        feat_list = [
            (column.feature_importance, column.column_display_name)
            for column in result.tables.tables_model_column_info
        ]
        feat_list.sort(reverse=True)
        if len(feat_list) < 10:
            feat_to_show = len(feat_list)
        else:
            feat_to_show = 10

        print("Features of top importance:")
        for feat in feat_list[:feat_to_show]:
            print(feat)