import requests
import pandas as pd 
import os
api_url = "https://api.github.com/repos/aavail/ai-workflow-capstone/contents/cs-train"

response = requests.get(api_url)
all_json_data= {}

if response.status_code ==200 :
    print("fetching data from github repo")
    files=response.json()
    for file in files :
        if file["type"] == "file" and file["name"].endswith(".json"):
            file_name= file["name"]
            print (f"file name is :{file_name}")
            download_url = file["download_url"]
            # Fetch the actual JSON file data
            file_response = requests.get(download_url)
            if file_response.status_code == 200:
                all_json_data[file_name] = file_response.json()
            else :
                print(f"Failed to fetch data for file: {file_name}")
        elif file["name"].endswith(".csv"):
            file_name= file["name"]
            print (f"file name is :{file_name}")
            download_url = file["download_url"]
            all_json_data[file_name] = pd.read_csv(download_url)
            print(f"Loaded CSV into DataFrame:  {file_name}")
    
                          
else :
    print("Failed to fetch file all  list from GitHub")


master_df = pd.concat([pd.DataFrame(v) for v in all_json_data.values()], ignore_index=True)

# Now save it
master_df=master_df.to_csv("all_training_data1.csv", index=False)
print("Saved combined dataset successfully!")

print("Your file is saved exactly here:")
print(os.path.abspath("all_training_data1.csv"))

#master_df.head()
#master_df.info()

#print(master_df.isnull().sum())

#for col in master_df.columns:
#    num_uniques = master_df[col].nunique()
  #  print(f"Column: {col}")
  #  print(f"  - Number of unique values: {num_uniques}")
