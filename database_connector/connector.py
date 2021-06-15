import paramiko as pm
import paramiko as pm
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv("env_file/.env_credentials")

REMOTE_HOST = os.environ.get("REMOTE_HOST")
REMOTE_USER = os.environ.get("REMOTE_USER")
REMOTE_PASSWORD = os.environ.get("REMOTE_PASSWORD")

def read_remote_df(remote_path, sep = ",", index_col = None, usecols = None):    
    if isinstance(remote_path, (str)):
        try:
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
    
    
def download_file_remote(remote_path, local_path):
    if isinstance(remote_path, (str)):
        try:
            local_dir ="/".join(local_path.split("/")[0:-1])
            if os.path.isdir(local_dir) == False :
                os.makedirs(local_dir )
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            sftp_client.get(remote_path, local_path)
            sftp_client.close()
        except:
            print("An error occurred.")
            sftp_client.close()

   
def download_dir_remote(remote_path, local_path):
    
    def list_files(startpath):
        list_arbotum = []
        for root, dirs, files in os.walk(startpath):
            for f in files:
                list_arbotum.append('{}/{}'.format(root, f))
        return list_arbotum
    if isinstance(remote_path, (str)):
        try:
            if os.path.isdir(local_path) == False :
                os.makedirs(local_path)
            for path in list_files(remote_path):
                if os.path.isdir("/".join(path.split("/")[4:-1])) == False :
                    os.makedirs("/".join(path.split("/")[4:-1]))
            client = pm.SSHClient()
            client.load_system_host_keys()
            client.connect(hostname = REMOTE_HOST,username = REMOTE_USER, password = REMOTE_PASSWORD)
            sftp_client = client.open_sftp()
            for path in list_files(remote_path):
                a=1
                #print("/".join(path.split("/")[4:-1]))
                #os.makedirs("/".join(path.split("/")[4:-1]))
                print(os.path.join(local_path,"/".join(path.split("/")[5:-1]),path.split("/")[-1]))
                sftp_client.get(path,os.path.join(local_path,path))
            sftp_client.close()
        except:
            print("An error occurred.")
            sftp_client.close()