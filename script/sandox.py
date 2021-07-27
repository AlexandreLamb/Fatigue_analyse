import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation.check_dataset import check_landmarks_len
from data_manipulation import DataFormator
import pandas as pd
from database_connector import  SFTPConnector
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_IRBA_DATA_VAS = os.environ.get("PATH_TO_IRBA_DATA_VAS")
sftp = SFTPConnector()

df = sftp.read_remote_df(PATH_TO_IRBA_DATA_VAS, index_col="Subject", usecols=["Subject", "Fatigue (Apres PVT)", "Fatigue (Avant PVT)"])

df.loc[lambda df: df["Fatigue (Apres PVT)"].between(0, 25), "Fatigue (Apres PVT)"] = 0
df.loc[lambda df: df["Fatigue (Apres PVT)"].between(25, 50), "Fatigue (Apres PVT)"] = 1
df.loc[lambda df: df["Fatigue (Apres PVT)"].between(50, 75), "Fatigue (Apres PVT)"] = 2
df.loc[lambda df: df["Fatigue (Apres PVT)"].between(75, 100), "Fatigue (Apres PVT)"] = 3

df.loc[lambda df: df["Fatigue (Avant PVT)"].between(0, 25), "Fatigue (Avant PVT)"] = 0
df.loc[lambda df: df["Fatigue (Avant PVT)"].between(25, 50), "Fatigue (Avant PVT)"] = 1
df.loc[lambda df: df["Fatigue (Avant PVT)"].between(50, 75), "Fatigue (Avant PVT)"] = 2
df.loc[lambda df: df["Fatigue (Avant PVT)"].between(75, 100), "Fatigue (Avant PVT)"] = 3
print(df)
