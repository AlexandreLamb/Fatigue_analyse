import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_measure(path_to_df, num_sec, fps):
    df = pd.read_csv(path_to_df, index_col=0)
    measure_list = [measure for measure in list(df) if measure != "frame"]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list + [measure +"_roc" for measure in list(df) if measure != "frame"])
    for measure in measure_list:
        print(int(len(df.index)/(num_sec*fps)))
        for index in range(0,int(len(df.index)/(num_sec*fps))) :
            df_plot.loc[index, measure] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].mean()
            df_plot.loc[index, measure + "_std"] = df[df["frame"].between( df["frame"][index * (num_sec*fps)],  df["frame"][(index+1) * (num_sec*fps)] )][measure].std()    
            
            
        plt.subplot(2,1,1)
        
        plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))])
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):])
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel(measure)
        plt.title(measure + " by "+ str(num_sec) + " sec per point")
        
        plt.subplot(2,1,2)
        
        plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))].pct_change()*100)
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):].pct_change()*100)     
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel("percent of change")
        plt.title("Rate of change of " + measure + " by "+ str(num_sec) + " sec per point")
        
        plt.show()        

def plot_pred(path_to_df, num_sec, fps):
    df = pd.read_csv(path_to_df, index_col=0)
    print(df)
    measure_list = [measure for measure in list(df)]
    df[measure_list] = (df[measure_list]-df[measure_list].min())/(df[measure_list].max()-df[measure_list].min())
    df_plot = pd.DataFrame(columns=measure_list)
    for measure in measure_list :
        for index in range(0,int(len(df.index)/(num_sec*fps))) :
            if measure == "target_pred_mean" or measure == "target_pred_max" :
                df_plot.loc[index, measure] =  df.loc[index * (num_sec*fps):(index+1) * (num_sec*fps)][measure].sum()
            else :
                df_plot.loc[index, measure] =  df.loc[index * (num_sec*fps):(index+1) * (num_sec*fps)][measure].mean()
        
        print(df_plot)
        """plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))], yerr=df_plot[measure][:int(10/(num_sec/60))].pct_change())
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):], yerr=df_plot[measure][-int(10/(num_sec/60)):].pct_change())
        """
        plt.errorbar(np.arange(10/(num_sec/60)), df_plot[measure][:int(10/(num_sec/60))], yerr=df_plot[measure][:int(10/(num_sec/60))].pct_change())
        plt.errorbar(np.arange(45/(num_sec/60),45/(num_sec/60)-10/(num_sec/60), step=-1), df_plot[measure][-int(10/(num_sec/60)):], yerr=df_plot[measure][-int(10/(num_sec/60)):].pct_change())
        
        
        plt.xlabel(str(num_sec) + " sec / point")
        plt.ylabel(measure)
        if measure == "target_pred" :
            plt.title("sum of "+ measure + " by min with ROC (rate of change)")
        else :
            plt.title(measure + " by min with ROC (rate of change)")
        plt.show()  
#plot_measure("data/stage_data_out/dataset_non_temporal/Irba_40_min/DESFAM-F_H92_VENDREDI/DESFAM-F_H92_VENDREDI.csv", 30,10)
plot_pred("data/stage_data_out/predictions/pred.csv", 30,10)    

"""
afficher : mean , std, min, max, variance sur un lapse de temps donné de

variation des paramètres sur n (15, 30) secondes prit en moyenne max et min sur les 10 minutes
   
afficher 2 graphs 
    
    
    
"""