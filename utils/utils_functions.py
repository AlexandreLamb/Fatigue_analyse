import pandas as pd

def make_landmarks_header():
    csv_header = []
    for i in range(1,69):
        csv_header.append("landmarks_"+str(i)+"_x")
        csv_header.append("landmarks_"+str(i)+"_y")
    return csv_header

def parse_path_to_name(path):
    name_with_extensions = path.split("/")[-1]
    name = name_with_extensions.split(".")[0]
    return name

def paths_to_df(csv_array):
    df_array = []
    for path in csv_array:
        df_array.append(pd.read_csv(path).rename(columns={"Unnamed: 0" : "frame"}))
    return df_array

def generate_columns_name(windows):
    columns_name = []
    for i in range(1,windows+1):
        columns_name.append("t_"+str(i))
    return columns_name

