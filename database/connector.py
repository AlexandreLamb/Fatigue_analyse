import paramiko as pm
import paramiko as pm
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv("env_file/.env")

REMOTE_HOST = os.environ.get("REMOTE_HOST")
REMOTE_USER = os.environ.get("REMOTE_USER")
REMOTE_PASSWORD = os.environ.get("REMOTE_PASSWORD")
REMOTE_DATA_URI = os.environ.get("REMOTE_DATA_URI")
PATH_TO_HDD_VIDEO_FOLDER_1 = os.environ.get("PATH_TO_HDD_VIDEO_FOLDER_1")
PATH_TO_HDD_VIDEO_FOLDER_2 = os.environ.get("PATH_TO_HDD_VIDEO_FOLDER_2")
user = os.environ.get("USERNAM")
print(user)
  
def read_remote_df(remote_path):    
    if isinstance(remote_path, (str)):
        try:
            print(os.path.join(REMOTE_DATA_URI,remote_path))
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            remote_file = sftp_client.open(os.path.join(REMOTE_DATA_URI,remote_path))
            df = pd.read_csv(remote_file)
            remote_file.close()
            sftp_client.close()
            return df 
        except:
            print("An error occurred.")
            sftp_client.close()
            remote_file.close()
    else:
        raise ValueError("Path to remote file as string required")
  
def save_remote_df(remote_path, df):    
    if isinstance(remote_path, (str)):
        
        print(os.path.join(REMOTE_DATA_URI,remote_path))
        client = pm.SSHClient()
        client.load_system_host_keys()
        client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
        sftp_client = client.open_sftp()
        with sftp_client.open(os.path.join(REMOTE_DATA_URI,remote_path), "w") as f:
            f.write(df.to_csv())
        sftp_client.close()
            
    else:
        raise ValueError("Path to remote file as string required")  
  