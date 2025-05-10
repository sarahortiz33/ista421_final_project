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
    """
    This function finds the frequency of each category of the features and
    response variable.

    col: A column from the Dataframe created from the
    cybersecurity_incidents.csv file

    return: A dictionary with the category as the key, and frequency as the
    value.
    """
    freq = {}
    for i in col:
        freq[i] = freq.get(i, 0) + 1
    return freq


country_freq = find_freq(country)
industry_freq = find_freq(industry)
attacks_freq = find_freq(attacks)


def eda_hist():
    plt.bar(list(country_freq.keys()), list(country_freq.values()), color='g', label='Countries')
    plt.bar(list(industry_freq.keys()), list(industry_freq.values()), color='orange', label='Industries')
    plt.bar(list(attacks_freq.keys()), list(attacks_freq.values()), color='b', label='Attack type')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Multiple Variables')

    plt.xticks(rotation=45)
    plt.xlim(-0.5)
    plt.tight_layout()

    plt.legend(loc='upper right')











def main():
    eda_hist()
    plt.show()


if __name__ == '__main__':
    main()
