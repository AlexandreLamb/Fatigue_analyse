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
                
                
    def plot_by_subject(self, subject, periode, num_min):
        subject_serie_rt = self.serie_pvt_rt_total[(self.serie_pvt_rt_total.index.get_level_values('sujet') == subject) & (self.serie_pvt_rt_total.index.get_level_values('periode') == periode)]
        subject_serie_lapse = self.serie_pvt_lapse_total[(self.serie_pvt_lapse_total.index.get_level_values('sujet') == subject) & (self.serie_pvt_lapse_total.index.get_level_values('periode') == periode)]
        plt.subplot(2,1,1)
        print(pd.DataFrame(subject_serie_rt.values[0][:num_min])[0].pct_change())
        plt.errorbar(np.arange(num_min) , subject_serie_rt.values[0][:num_min], yerr=pd.DataFrame(subject_serie_rt.values[0][:num_min])[0].pct_change()*100)
        plt.errorbar(np.arange(len(list(subject_serie_rt)),len(list(subject_serie_rt))-num_min, -1) , subject_serie_rt.values[0][-num_min:], yerr= pd.DataFrame(subject_serie_rt.values[0][-num_min:])[0].pct_change()*100)
        plt.xlabel("min")
        plt.ylabel("response time")
        plt.title(subject_serie_lapse.index)
        plt.subplot(2,1,2)
        plt.errorbar(np.arange(num_min) , subject_serie_lapse.values[0][:num_min])
        plt.errorbar(np.arange(len(list(subject_serie_lapse)),len(list(subject_serie_lapse))-num_min, -1) , subject_serie_lapse.values[0][-num_min:])
        plt.xlabel("min")
        plt.ylabel("lapse")
        #plt.title(list(subject_serie_lapse.index))
        
        plt.show()
pvt = PvtReader("data/stage_data_out/sujets_data_pvt_perf.csv")
pvt.plot_by_subject("H92", "T1", 10)