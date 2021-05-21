import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

class PvtReader():
    def __init__(self, path_to_df_pvt):
        self.df_pvt_total = pd.read_csv(path_to_df_pvt, sep=";", index_col = [0,1,2])
        self.serie_pvt_rt_total = self.df_pvt_total.filter(regex=("rt_")).iloc[:,12:]
        self.serie_pvt_lapse_total = self.df_pvt_total.filter(regex=("lapse_"))
    """                 
        
        self.df_pvt_T1_dette = self.df_pvt_total.loc[(self.df_pvt_total.index.get_level_values('condition') == "dette") & (self.df_pvt_total.index.get_level_values('periode') == "T1")]
        self.df_pvt_T2_dette = self.df_pvt_total.loc[(self.df_pvt_total.index.get_level_values('condition') == "dette") & (self.df_pvt_total.index.get_level_values('periode') == "T2")]
        self.df_pvt_T1_temoin = self.df_pvt_total.loc[(self.df_pvt_total.index.get_level_values('condition') == "temoin") & (self.df_pvt_total.index.get_level_values('periode') == "T1")]
        self.df_pvt_T2_temoin = self.df_pvt_total.loc[(self.df_pvt_total.index.get_level_values('condition') == "temoin") & (self.df_pvt_total.index.get_level_values('periode') == "T2")]

        self.serie_rt_T1_dette = self.df_pvt_T1_dette.filter(regex=("rt_")).iloc[:,12:]
        self.serie_rt_T2_dette = self.df_pvt_T2_dette.filter(regex=("rt_")).iloc[:,12:]
        self.serie_rt_T1_temoin = self.df_pvt_T1_temoin.filter(regex=("rt_")).iloc[:,12:]
        self.serie_rt_T2_temoin = self.df_pvt_T2_temoin.filter(regex=("rt_")).iloc[:,12:]

        self.serie_lapse_T1_dette = self.df_pvt_T1_dette.filter(regex=("lapse_"))
        self.serie_lapse_T2_dette = self.df_pvt_T2_dette.filter(regex=("lapse_"))
        self.serie_lapse_T1_temoin = self.df_pvt_T1_temoin.filter(regex=("lapse_"))
        self.serie_lapse_T2_temoin = self.df_pvt_T2_temoin.filter(regex=("lapse_"))
    """
    def plot(self):
        for i in range(0,len(self.serie_rt_T1_dette)):
            self.serie_rt_T1_dette.iloc[i].plot(legend = self.serie_rt_T1_dette.iloc[i].name)
        plt.show()
        for i in range(0,len(self.serie_rt_T2_dette)):
            self.serie_rt_T2_dette.iloc[i].plot(legend = self.serie_rt_T2_dette.iloc[i].name)
        plt.show()
        for i in range(0,len(self.serie_rt_T1_temoin)):
            self.serie_rt_T1_temoin.iloc[i].plot(legend = self.serie_rt_T1_temoin.iloc[i].name)
        plt.show()    
        for i in range(0,len(self.serie_rt_T2_temoin)):
            self.serie_rt_T2_temoin.iloc[i].plot(legend = self.serie_rt_T2_temoin.iloc[i].name) 
        plt.show()
            
        for i in range(0,len(self.serie_lapse_T1_dette)):
            self.serie_lapse_T1_dette.iloc[i].plot(kind ="hist" ,legend = self.serie_lapse_T1_dette.iloc[i].name)          
        plt.show()
        for i in range(0,len(self.serie_lapse_T2_dette)):
            self.serie_lapse_T2_dette.iloc[i].plot(kind ="hist" ,legend = self.serie_lapse_T2_dette.iloc[i].name)      
        plt.show()
        for i in range(0,len(self.serie_lapse_T1_temoin)):
            self.serie_lapse_T1_temoin.iloc[i].plot(kind ="hist" ,legend = self.serie_lapse_T1_temoin.iloc[i].name)
        plt.show()
        for i in range(0,len(self.serie_lapse_T2_temoin)):
            self.serie_lapse_T2_temoin.iloc[i].plot(kind ="hist" ,legend = self.serie_lapse_T2_temoin.iloc[i].name)
        plt.show()
                
                
    def plot_by_subject(self, num_min = 0):
        for subject in self.serie_pvt_rt_total.groupby(level=[0,1,2]):
            #fig, axs = plt.subplots(2,1)
            subject_index = "_".join(subject[0])
            subject_title =  " ".join(subject[0])
            """if num_min == 0:
                axs[0].errorbar(np.arange(len(subject[1].values[0])),subject[1].values[0])
                axs[1].errorbar(np.arange(len(subject[1].values[0])),pd.DataFrame(subject[1].values[0])[0].pct_change()*100)
            else : 
                axs[0].errorbar(np.arange(num_min) , subject[1].values[0][:num_min])
                axs[0].errorbar(np.arange(len(list(subject[1])),len(list(subject[1]))-num_min, -1) , subject[1].values[0][-num_min:])
                axs[1].errorbar(np.arange(num_min),pd.DataFrame(subject[1].values[0][:num_min])[0].pct_change()*100)
                axs[1].errorbar(np.arange(len(list(subject[1])),len(list(subject[1]))-num_min, -1), pd.DataFrame(subject[1].values[0][-num_min:])[0].pct_change()*100)
            
            axs[0].set_xlabel("min")
            axs[0].set_ylabel("response time")
            axs[0].set_title(subject_title)
             
            axs[1].set_xlabel("min")
            axs[1].set_ylabel("rate of chagne (%)")
            
            path_folder_to_save = "data/stage_data_out/resutls/img/"+subject_index+"/pvt/"
            path_img_to_save = path_folder_to_save +"pvt.png"
            if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
            fig = plt.gcf()
            fig.set_size_inches((22, 10), forward=False)
            fig.savefig(path_img_to_save, dpi=500) 
            plt.close()
            """
        for subject in self.serie_pvt_lapse_total.groupby(level=[0,1,2]):
            subject_index = "_".join(subject[0])
            subject_title =  " ".join(subject[0])
            fig, axs = plt.subplots(1,1)
            if num_min == 0:
                axs.errorbar(np.arange(len(subject[1].values[0])),subject[1].values[0])
            else :   
                axs.errorbar(np.arange(len(list(subject[1])),len(list(subject[1]))-num_min, -1) , subject[1].values[0][-num_min:])
            axs.set_xlabel("min")
            axs.set_ylabel("response time")
            axs.set_title(subject_title)
        
            path_folder_to_save = "data/stage_data_out/resutls/img/"+subject_index+"/lapse/"
            path_img_to_save = path_folder_to_save +"lapse.png"
            if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
            fig = plt.gcf()
            fig.set_size_inches((22, 10), forward=False)
            fig.savefig(path_img_to_save, dpi=500) 
            plt.close()

pvt = PvtReader("data/stage_data_out/sujets_data_pvt_perf.csv")
pvt.plot_by_subject()

