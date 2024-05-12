import requests

# URL of the dataset
url = "https://www.kaggle.com/datasets/ajetkharbri/romania-aqi-2022-1-to-2023-5"

# Filename for saving the dataset
output_file = "aqi.csv"

# Send a GET request to download the dataset
response = requests.get(url)

# Save the dataset to a file
with open(output_file, "wb") as file:
    file.write(response.content)



