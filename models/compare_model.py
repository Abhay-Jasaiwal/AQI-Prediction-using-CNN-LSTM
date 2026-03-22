import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 Manually enter results after running models
results = [
    ["CNN", "Dwarka", 15.259533721312783, 10.427746263758445, -0.2637826897458633],
    ["LSTM", "Dwarka", 14.004375983699086, 8.177711405255659, -0.0644307197552898],
    ["CNN-LSTM", "Dwarka", 14.394851042649794, 9.577683722833083, -0.12461591788612525],
    ["CNN", "Rohini", 10.524615680367871, 8.549163730968601, -0.30362224684606676],
    ["LSTM", "Rohini", 14.95451264535633, 13.801257719619997, -1.6319873640135536],
    ["CNN-LSTM", "Rohini", 21.102215546726605, 19.404875986032376, -4.240773384787964],
]

df = pd.DataFrame(results, columns=["Model", "Dataset", "RMSE", "MAE", "R2"])

print(df)

plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="RMSE", hue="Dataset", data=df)
plt.title("Model Comparison (RMSE)")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="R2", hue="Dataset", data=df)
plt.title("Model Comparison (R2 Score)")
plt.show()