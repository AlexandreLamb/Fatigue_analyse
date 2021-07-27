import pandas as pd

df_metrics_model_train = pd.DataFrame(columns=["binary_accuracy","binary_crossentropy","mean_squared_error"])

df_metricsmodel_train = df_metrics_model_train.append({"binary_accuracy":1,"binary_crossentropy":1,"mean_squared_error":1}, ignore_index=True)

print(df_metrics_model_train)