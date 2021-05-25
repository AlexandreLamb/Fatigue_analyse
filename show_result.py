import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg', force=True)
import numpy as np
from pandas.plotting import table
import os
from utils import parse_video_name

def plot_measure(path_to_df, num_sec, fps, subject):
    df = pd.read_csv(path_to_df, index_col=0)
    measure_list = [measure for measure in list(df) if measure != "frame"]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list + [measure +"_roc" for measure in list(df) if measure != "frame"])
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
    df = pd.read_csv(path_to_df, index_col=0)
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
        

def generate_data_img(csv_folder = "data/stage_data_out/dataset_non_temporal/Irba_40_min", num_sec_to_test = [1,3,5,10,20,30,40,60], fps =10):
    list_subject = os.listdir(csv_folder)
    for num_sec in num_sec_to_test:
        for subject in list_subject:
            plot_measure(csv_folder + "/" + subject + "/" + subject +".csv", num_sec, fps, subject)
            plot_pred("data/stage_data_out/predictions/pred.csv", num_sec, fps, subject)
            #sprint(subject + " plot is save in data/stage_data_out/resutls/img/"+subject+"/"+str(num_sec))


generate_data_img()
"""
afficher : mean , std, min, max, variance sur un lapse de temps donné de

variation des paramètres sur n (15, 30) secondes prit en moyenne max et min sur les 10 minutes
   
afficher 2 graphs 
    
    
    
"""
