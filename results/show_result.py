from numpy.lib import index_tricks
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#matplotlib.use('TkAgg', force=True)
import numpy as np
from pandas.plotting import table
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tensorflow.python.ops.gen_io_ops import reader_read
from utils import parse_video_name
import time
from dotenv import load_dotenv
load_dotenv("env_file/.env")
from database_connector import read_remote_df, save_remote_df, list_dir_remote
PATH_TO_RESULTS = os.environ.get("PATH_TO_RESULTS")
PATH_TO_RESULTS_CROSS_PREDICTIONS = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS")
PATH_TO_RESULTS_PREDICTIONS = os.environ.get("PATH_TO_RESULTS_PREDICTIONS")

def plot_measure(path_to_df, num_sec, fps, subject):
    df = read_remote_df(path_to_df, index_col=0)
    measure_list = [measure for measure in list(df) if measure != "frame"]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list + [measure +"_roc" for measure in list(df) if measure != "frame"])
    df_stat =  pd.DataFrame(columns=["subject","measure","target","mean", "std","min","max"]).set_index(["subject","measure","target"])
    print(df_stat)
    for measure in measure_list:
        for index in range(0,int(len(df.index)/(num_sec*fps))-1) :
            df_plot.loc[index, measure] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].mean()
            df_plot.loc[index, measure + "_std"] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].std()    
        
        df_1 = df_plot[measure][:int(5/(num_sec/60))]
        df_2 = df_plot[measure][-int(5/(num_sec/60)):]
        fig, axs = plt.subplots(3,1)
        
        axs[0].errorbar(np.arange(5/(num_sec/60)), df_plot[measure][:int(5/(num_sec/60))])
        axs[0].errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-5/(num_sec/60), step=-1), df_plot[measure][-int(5/(num_sec/60)):])
        axs[0].set_ylabel(measure)
        axs[0].set_title(measure + " by "+ str(num_sec) + " sec per point")
    
        axs[1].errorbar(np.arange(5/(num_sec/60)), df_plot[measure][:int(5/(num_sec/60))].pct_change()*100)
        axs[1].errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-5/(num_sec/60), step=-1), df_plot[measure][-int(5/(num_sec/60)):].pct_change()*100)     
        axs[1].set_xlabel(str(num_sec) + " sec / point")
        axs[1].set_ylabel("percent of change")
        axs[1].set_title("Rate of change of " + measure + " by "+ str(num_sec) + " sec per point")
        
        measure_describe_1 = pd.DataFrame(df_1.astype("float32").describe().round(3)).transpose().drop("count", axis=1)
        measure_describe_1 = measure_describe_1.rename(index={measure : measure + " of 5 first min"})
        
        measure_describe_2 = pd.DataFrame(df_2.astype("float32").describe().round(3)).transpose().drop("count", axis=1)
        measure_describe_2 = measure_describe_2.rename(index={measure : measure + " of 5 last min"})
        
        roc_describe_1 = pd.DataFrame((df_1.pct_change()*100).describe().round(3)).transpose().drop("count", axis=1)
        roc_describe_1 = roc_describe_1.rename(index={measure : measure + " ROC of 5 first min"})
        
        roc_describe_2 = pd.DataFrame((df_2.pct_change()*100).describe().round(3)).transpose().drop("count", axis=1)
        roc_describe_2 = roc_describe_2.rename(index={measure : measure + " ROC of 5 last min"})
        
        axs[2].axis("off")
        tabla = table(axs[2], measure_describe_1.append([measure_describe_2, roc_describe_1, roc_describe_2]), loc='best', colWidths=[0.1]*len(measure_describe_1.columns))
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(12) 
        tabla.scale(1.2, 1.2)
        
        path_folder_to_save = "data/stage_data_out/resutls/img/"+parse_video_name([subject])[0]+"/"+str(num_sec)+"sec_threeshold/measures/"
        path_img_to_save = path_folder_to_save + measure +".png"
      
        if os.path.exists(path_folder_to_save) == False:
            os.makedirs(path_folder_to_save)
        fig = plt.gcf()
        fig.set_size_inches((22, 10), forward=False)
        fig.savefig(path_img_to_save, dpi=500) 
        plt.close()       
def plot_pred(path_to_df, num_sec, fps, subject):
    """function who plot the prediction of a model 

    Args:
        path_to_df ([type]): [description]
        num_sec ([type]): [description]
        fps ([type]): [description]
        subject ([type]): [description]
    """
    df = read_remote_df(path_to_df, index_col=0)
    measure_list = [measure for measure in list(df)]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list)
    for measure in measure_list :
        for index in range(0,int(len(df.index)/(num_sec*fps))) :
            if measure == "target_pred_mean" or measure == "target_pred_max" :
                df_plot.loc[index, measure] =  df.loc[index * (num_sec*fps):(index+1) * (num_sec*fps)][measure].sum()
            else :
                df_plot.loc[index, measure] =  df.loc[index * (num_sec*fps):(index+1) * (num_sec*fps)][measure].mean()
        
        plt.subplot(2,1,1)
        
        plt.errorbar(np.arange(5/(num_sec/60)), df_plot[measure][:int(5/(num_sec/60))])
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-5/(num_sec/60), step=-1), df_plot[measure][-int(5/(num_sec/60)):])
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel(measure)
        plt.title(measure + " prediciton by "+ str(num_sec) + " sec per point")
        
        plt.subplot(2,1,2)
        
        plt.errorbar(np.arange(5/(num_sec/60)), df_plot[measure][:int(5/(num_sec/60))].pct_change()*100)
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-5/(num_sec/60), step=-1), df_plot[measure][-int(5/(num_sec/60)):].pct_change()*100)     
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel("percent of change")
        plt.title("Rate of change of " + measure + " prediction by "+ str(num_sec) + " sec per point")
        
        path_folder_to_save = "data/stage_data_out/resutls/img/"+parse_video_name([subject])[0]+"/"+str(num_sec)+"sec_threeshold/predictions/"
        path_img_to_save = path_folder_to_save + measure +".png"
        print("img save into : " + path_img_to_save)
        if os.path.exists(path_folder_to_save) == False:
            os.makedirs(path_folder_to_save)
        fig = plt.gcf()
        fig.set_size_inches((22, 10), forward=False)
        fig.savefig(path_img_to_save, dpi=500) 
        plt.savefig(path_img_to_save)  
        plt.close()
    
def generate_mean_diff(csv_folder = "data/stage_data_out/dataset_non_temporal/Irba_40_min", num_sec_list=[1,3,5,10,60], fps =10):
    list_subject = list_dir_remote(csv_folder)
    df_stat =  pd.DataFrame(columns=["subject","windows_size","measure","target","mean", "std","min","max"]).set_index(["subject","windows_size","measure","target"])

    for subject in list_subject:
        df = read_remote_df(csv_folder + "/" + subject + "/" + subject +".csv", index_col=0)
        measure_list = [measure for measure in list(df) if measure != "frame"]
        df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
        df_plot = pd.DataFrame(columns=measure_list + [measure +"_roc" for measure in list(df) if measure != "frame"])
        for num_sec in num_sec_list:
            for measure in measure_list: 
                for index in range(0,int(len(df.index)/(num_sec*fps))-1) :
                    df_plot.loc[index, measure] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].mean()
                    df_plot.loc[index, measure + "_std"] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].std()    
            
                df_1 = df_plot[measure][:int(5/(num_sec/60))]
                df_2 = df_plot[measure][-int(5/(num_sec/60)):]
                measure_describe_1 = pd.DataFrame(df_1.astype("float32").describe().round(3)).transpose().drop("count", axis=1)
                measure_describe_2 = pd.DataFrame(df_2.astype("float32").describe().round(3)).transpose().drop("count", axis=1)
                df_stat.loc[(subject, num_sec ,measure, "non_fatigue"),["mean","std","min","max"]] = measure_describe_1[["mean","std","min","max"]].values[0]
                df_stat.loc[(subject, num_sec ,measure, "fatigue"),["mean","std","min","max"]] = measure_describe_2[["mean","std","min","max"]].values[0]
                df_stat.loc[(subject, num_sec ,measure, "diff abs"),["mean","std","min","max"]] = np.absolute(np.subtract(measure_describe_1[["mean","std","min","max"]].values[0], measure_describe_2[["mean","std","min","max"]].values[0]))
            #print(df_stat)
    path_to_csv_stat = os.path.join(PATH_TO_RESULTS, "statistiques.csv")
    save_remote_df(path_to_csv_stat, df_stat)
def generate_data_img(csv_folder = "data/stage_data_out/dataset_non_temporal/Irba_40_min", num_sec_to_test = [1,3,5,10,60], fps =10):
    list_subject = list_dir_remote(csv_folder)
    for num_sec in num_sec_to_test:
        for subject in list_subject:
            plot_measure(csv_folder + "/" + subject + "/" + subject +".csv", num_sec, fps, subject)
            #plot_pred("data/stage_data_out/predictions/pred.csv", num_sec, fps, subject)
            #sprint(subject + " plot is save in data/stage_data_out/resutls/img/"+subject+"/"+str(num_sec))

#generate_mean_diff()

#generate_data_img()
"""
afficher : mean , std, min, max, variance sur un lapse de temps donné de

variation des paramètres sur n (15, 30) secondes prit en moyenne max et min sur les 10 minutes
   
afficher 2 graphs 
    
    
    
"""
def exctract_pvt_hours_from_eva_file():
    file_list =  list_dir_remote("data/stage_data_out/eva_data/Resultat_EVA")
    df_pvt_hours = pd.DataFrame(columns=["subject", "moment","date", "hours"]).set_index(["subject", "date","moment"])
    for index, file in enumerate(file_list):
        file_split = file.split("_")
        subject = file_split[4]
        moment = file_split[2]
        date = file_split[5]
        hours = file_split[-1].split(".")[0]
        df_pvt_hours.loc[(subject,date,moment+"_pvt"),["hours"]] = hours
    df_pvt_hours.sort_index().to_excel("data/stage_data_out/pvt_hours_by_subjects.xlsx")  

    
def show_cross_validation_resutlt():
    path_to_cross_folder = PATH_TO_RESULTS_CROSS_PREDICTIONS
    df_metrics = pd.DataFrame()
    df_pred = pd.DataFrame()
    dict_pred = {}
    for video_name in [path for path in list_dir_remote(path_to_cross_folder)]:
        dict_pred[video_name] = {}
        for sub_path in [os.path.join(path_to_cross_folder , video_name, sub_path) for sub_path in list_dir_remote(path_to_cross_folder + video_name)] : 
           if os.path.isdir(sub_path) :
                path_arr_file = [os.path.join(sub_path, file) for file in list_dir_remote(sub_path)]
                for file in path_arr_file:
                    if "metrics" in file:
                        df_metrics = df_metrics.append(read_remote_df(file, index_col=[0,1]))
                    elif "pred" in file:
                        df_csv = read_remote_df(file, usecols=["pred_mean","pred_max","target_pred_mean","target_pred_max","target_real"])
                        measure_combination = file.split("/")[-2]
                        video_exclude = file.split("/")[-3]
                        
                        dict_pred[video_exclude][measure_combination] =  df_csv
                        #print(confusion_matrix(df_csv["target_real"],df_csv["target_pred_mean"]))
                        #df_pred = df_pred.append(df_csv)
    print(df_metrics.loc["DESFAM_F_H95_VENDREDI"].sort_values("binary_accuracy", ascending=False))
    intersting_mertics = df_metrics[df_metrics["binary_accuracy"]>= 0.85].sort_values("binary_accuracy", ascending=False).index.values
    #print(df_metrics.sort_values("binary_accuracy", ascending=False))
    """for metrics in intersting_mertics:
        print(metrics[0],metrics[1])
        print(confusion_matrix(dict_pred[metrics[0]][metrics[1]]["target_real"],dict_pred[metrics[0]][metrics[1]]["target_pred_mean"]))
    """
    #print(dict_pred["DESFAM_F_H95_VENDREDI"])
    #print(df_metrics.)



def show_pred_result_video():
    df_pred = read_remote_df(os.path.join(PATH_TO_RESULTS_PREDICTIONS+"DESFAM_F_H95_VENDREDI","DESFAM_F_H95_VENDREDI_pred.csv"))
    windows_sec_0 = np.arange(0,50,10)*600
    windows_sec_5 = np.arange(5,50,10)*600
    print(df_pred)
    for index_inf, index_sup in zip(windows_sec_0, windows_sec_5):
        print(index_inf," to " ,index_sup)
        print("pred_mean",df_pred.loc[index_inf:index_sup]["pred_mean"].mean())
        #print("target_pred_mean",df_pred.loc[index_inf:index_sup]["target_pred_mean"].mean())
        #print("pred_max",df_pred.loc[index_inf:index_sup]["pred_max"].mean())
        #print("target_pred_max",df_pred.loc[index_inf:index_sup]["target_pred_max"].mean())
    
    
show_pred_result_video()
