import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cybersecurity_incidents.csv")

# Check for missing values in the dataset
for index, row in df.iterrows():
    for col in df.columns:
        if pd.isna(row[col]):
            print("Missing value at row:", index, " column", col)

# Features and response variable
country = df["country"]
industry = df["industry"]
attacks = df["attack_type"]


def find_freq(col):
    freq = {}
    for i in col:
        freq[i] = freq.get(i, 0)+1
    return freq


country_freq = find_freq(country)
industry_freq = find_freq(industry)
attacks_freq = find_freq(attacks)


plt.hist(country_freq, bins=30, label='Variable 1')
plt.hist(industry_freq, bins=30, label='Variable 2')
plt.hist(attacks_freq, bins=30, label='Variable 3')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Multiple Variables')
plt.legend(loc='upper right')


plt.show()

