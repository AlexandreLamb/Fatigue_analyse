import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

df["ear_l"] = (df["euclid_dist_38_42_l"]+df["euclid_dist_39_41_l"])/(2*df["euclid_dist_37_40_l"])
df["ear_r"] = (df["euclid_dist_44_48_r"]+df["euclid_dist_45_47_r"])/(2*df["euclid_dist_43_46_r"])
df["ear"] = (df["ear_l"]+df["ear_r"])/2

df[('ear')].plot()
plt.xlabel("frame")
plt.ylabel("eye aspect ratio")
plt.show()
