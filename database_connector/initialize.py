import os
from dotenv import load_dotenv
from connector import download_dir_remote, download_file_remote
load_dotenv("env_file/.env_credentials")


PATH_TO_ENV_PATH = os.environ.get("PATH_TO_ENV_PATH")
download_file_remote(PATH_TO_ENV_PATH, "env_file/.env_path")
load_dotenv("env_file/.env_path")
#PATH_TO_UTILS_FILE_TO_DOWNLOAD = os.environ.get("PATH_TO_UTILS_FILE_TO_DOWNLOAD")
#download_dir_remote("/home/fatigue_database/fatigue_database/landmarks/","utils_files/")

