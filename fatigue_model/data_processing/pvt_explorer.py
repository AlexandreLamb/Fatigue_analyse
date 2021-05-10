import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

class PvtReader():
    def __init__(self, path_to_df_pvt):
        self.df_pvt_total = pd.read_csv(path_to_df_pvt, sep=";", index_col = [0,1,2])
        self.df_pvt_rt_total = self.df
        
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
                
pvt = PvtReader("data/stage_data_out/sujets_data_pvt_perf.csv")
pvt.plot()