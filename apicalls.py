import json
import os
import pandas as pd
import requests

with open("config.json",'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
#Specify a URL that resolves to your workspace
URL = "http://localhost:8100/"

#Define the API endpoints
endpoints = {
    "first_run": "first_run_on_practice_data",
    "prediction": "prediction",
    "scoring": "scoring",
    "summarystats": "summarystats",
    "diagnostics": "diagnostics"
}

def call_prediction_api():
    # Load test data as a DataFrame and convert to JSON
    test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
    test_df = pd.read_csv(test_data_path)
    test_json = test_df.to_dict(orient='records')
    response = requests.post(f"{URL}{endpoints['prediction']}", json=test_json)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to call prediction endpoint", "status_code": response.status_code}

#Function to call an API endpoint
def call_api(endpoint):
    response = requests.get(f"{URL}{endpoints[endpoint]}")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to call {endpoint} endpoint", "status_code": response.status_code}


#Call each API endpoint and store the responses
response0 = call_api("first_run")
response1 = call_prediction_api()
response2 = call_api("scoring")
response3 = call_api("summarystats")
response4 = call_api("diagnostics")
#Combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

#Write the responses to your workspace apireturns.txt.
output_file_path = os.path.join(output_folder_path, "apireturns.txt")
with open(output_file_path, 'w') as f:
    for endpoint, response in responses.items():
        f.write(f"Response from {endpoint} endpoint:\n")
        f.write(json.dumps(response, indent=4))
        f.write("\n\n")

#Check if the output file was created successfully
if os.path.exists(output_file_path):
    print(f"API responses successfully written to {output_file_path}")



