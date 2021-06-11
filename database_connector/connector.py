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


def read_remote_df(remote_path, sep = ",", index_col = None, usecols = None):    
    if isinstance(remote_path, (str)):
        try:
            print(os.path.join(REMOTE_DATA_URI,remote_path))
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            remote_file = sftp_client.open(remote_path)
            df = pd.read_csv(remote_file, sep = sep , index_col = index_col, usecols = usecols)
            remote_file.close()
            sftp_client.close()
            return df 
        except:
            print("An error occurred.")
            sftp_client.close()
            remote_file.close()
    else:
        raise ValueError("Path to remote file as string required")
  
def save_remote_df(remote_path, df, index = True, mode='w', index_label = None, header = True):    
    if isinstance(remote_path, (str)):
        try: 
            print(os.path.join(REMOTE_DATA_URI,remote_path))
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            with sftp_client.open(remote_path, "w") as f:
                f.write(df.to_csv(index = index, mode = mode, index_label = index_label, header = header))
            sftp_client.close()
        except:
            print("An error occurred.")
            sftp_client.close()
            
    else:
        raise ValueError("Path to remote file as string required")  
    
def list_dir_remote(remote_path):
    if isinstance(remote_path, (str)):
        try:
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            list_dir = sftp_client.listdir(remote_path)
            sftp_client.close()
            return list_dir
        except:
            print("An error occurred.")
            sftp_client.close()
    else:
        raise ValueError("Path to remote file as string required")
  
