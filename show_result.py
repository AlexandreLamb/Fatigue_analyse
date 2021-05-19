import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def plot_measure(path_to_df, num_sec, fps, subject):
    df = pd.read_csv(path_to_df, index_col=0)
    measure_list = [measure for measure in list(df) if measure != "frame"]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list + [measure +"_roc" for measure in list(df) if measure != "frame"])
    for measure in measure_list:
        for index in range(0,int(len(df.index)/(num_sec*fps))) :
            df_plot.loc[index, measure] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].mean()
            df_plot.loc[index, measure + "_std"] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].std()    
            
            
        fig, axs = plt.subplots(2,1)
        
        axs[0].errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))])
        axs[0].errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):])
        axs[0].set_xlabel(str(num_sec) + " sec / point")
        axs[0].set_ylabel(measure)
        axs[0].set_title(measure + " by "+ str(num_sec) + " sec per point")
    
        axs[1].errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))].pct_change()*100)
        axs[1].errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):].pct_change()*100)     
        axs[1].set_xlabel(str(num_sec) + " sec / point")
        axs[1].set_ylabel("percent of change")
        axs[1].set_title("Rate of change of " + measure + " by "+ str(num_sec) + " sec per point")
        path_folder_to_save = "data/stage_data_out/resutls/img/"+subject+"/"+str(num_sec)+"/measures/"
        path_img_to_save = path_folder_to_save + measure +".png"
        #plt.table([["1","2"],["2","4"]], loc = "bottom")
        #plt.autoscale()
        plt.show()
        if os.path.exists(path_folder_to_save) == False:
            os.makedirs(path_folder_to_save)
        plt.savefig(path_img_to_save) 
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
        
        plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))])
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):])
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel(measure)
        plt.title(measure + " prediciton by "+ str(num_sec) + " sec per point")
        
        plt.subplot(2,1,2)
        
        plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))].pct_change()*100)
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):].pct_change()*100)     
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel("percent of change")
        plt.title("Rate of change of " + measure + " prediction by "+ str(num_sec) + " sec per point")
        
        path_folder_to_save = "data/stage_data_out/resutls/img/"+subject+"/"+str(num_sec)+"/predictions/"
        path_img_to_save = path_folder_to_save + measure +".png"
        if os.path.exists(path_folder_to_save) == False:
            os.makedirs(path_folder_to_save)
        plt.savefig(path_img_to_save)  
        plt.close()
        
        


plot_measure("data/stage_data_out/dataset_non_temporal/Irba_40_min/DESFAM-F_H92_VENDREDI/DESFAM-F_H92_VENDREDI.csv", 10,10, "DESFAM-F_H92_VENDREDI")
plot_pred("data/stage_data_out/predictions/pred.csv", 10,10, "DESFAM-F_H92_VENDREDI")    

"""
afficher : mean , std, min, max, variance sur un lapse de temps donné de

variation des paramètres sur n (15, 30) secondes prit en moyenne max et min sur les 10 minutes
   
afficher 2 graphs 
    
    
    
"""