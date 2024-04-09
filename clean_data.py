import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

important_cols = ["participant", "condition", "reward", "loss", "total", "mouse.clicked_name", 'mouse.time']

# getting all data files in an array.
data_files = []
folder_path = "./data/*.csv"
csv_files = glob.glob(folder_path)
data_files.extend(csv_files)

# checking the number of data files
print("no csv files found") if len(data_files) == 0 else print(f"found {len(data_files)} csv files")


for file in data_files:
    data = pd.read_csv(file)
    consolidated_data = pd.DataFrame(columns=important_cols)
    columns_to_extract = [col for col in important_cols if col in data.columns]
    if columns_to_extract:
        consolidated_data = pd.concat([consolidated_data, data[columns_to_extract]], ignore_index=True)
        consolidated_data.dropna(subset=["condition"], inplace=True)
        consolidated_data['mouse.clicked_name'] = consolidated_data['mouse.clicked_name'].str.extract(r"'(.*?)'")
        participant = consolidated_data['participant'].unique()[0]
        consolidated_data.to_csv(f"./cleaned_Data/{participant}_data.csv", index=False)
        print(f"{participant}_data.csv created")


# consolidated_data.to_csv(f"./{participant}_data.csv", index=False)

# # removing all the empty rows & clearning clickedName and time
# consolidated_data.dropna(subset=["condition"], inplace=True)
# consolidated_data['mouse.clicked_name'] = consolidated_data['mouse.clicked_name'].str.extract(r"'(.*?)'")
# participant = consolidated_data['participant'].unique()[0]


# consolidated_data.to_csv(f"./{participant}_data.csv", index=False)

