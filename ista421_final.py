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
severity = df["severity_level"]


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
severe_freq = find_freq(severity)


def eda_hist():
    """
    Plots the frequency of each category for the features and response variable.

    return: Nothing is returned by this function, as a plot is created.
    """
    plt.bar(list(country_freq.keys()), list(country_freq.values()), color='g', label='Countries')
    plt.bar(list(industry_freq.keys()), list(industry_freq.values()), color='orange', label='Industries')
    plt.bar(list(severe_freq.keys()), list(severe_freq.values()), color='b', label='Severity Level')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Multiple Variables')

    plt.xticks(rotation=45)
    plt.xlim(-0.5)
    plt.tight_layout()

    plt.legend(loc='upper right')


# One hot-encoded variables
df_dumb = pd.get_dummies(df, columns=['country', 'industry'], drop_first=True, dtype=int)

#print(df_dumb.columns)


response_vals = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
severity_num = []

for i in df_dumb["severity_level"]:
    if i == "Low":
        severity_num.append(response_vals["Low"])
    elif i == "Medium":
        severity_num.append(response_vals["Medium"])
    elif i == "High":
        severity_num.append(response_vals["High"])
    else:
        severity_num.append(response_vals["Critical"])








# need to model Low vs Critical
#               Medium vs Critical
#               High vs Critical
# log odds


def log_odds(intercept, change):
    pass
    # intercept
    # intercept + sum of each coefficient for severity level * the feature


# reference industry: Education
# reference country: Australia
# reference severity: Critical

def main():
    #eda_hist()
    plt.show()


if __name__ == '__main__':
    main()
