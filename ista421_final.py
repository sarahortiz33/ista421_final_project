import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# move to main
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

# print(df_dumb.columns)


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

X = df_dumb[['country_Brazil', 'country_Canada', 'country_China',
             'country_France', 'country_Germany', 'country_India',
             'country_Russia', 'country_UK', 'country_USA', 'industry_Finance',
             'industry_Government', 'industry_Healthcare',
             'industry_Manufacturing', 'industry_Retail', 'industry_Tech']]

y = np.array(severity_num)

X = X.copy()
X["Intercept"] = [1 for i in range(len(X))]
X_np = X.to_numpy()

n_obs = 1000
n_features = 1000

n_classes = 4
critical = 3
n_severity = n_classes - 1

# Set betas to 0
#betas = np.zeros((n_severity, X_np.shape[1]))

np.random.seed(42)
betas = np.random.randn(n_severity, X_np.shape[1])
# print(betas)


def log_odds(intercept, beta):
    """
    This function calculates the log odds for each category from the response
    variable vs the reference/baseline category, which is the "Critical"
    severity level.

    intercept: A 2D Numpy array that has the predictor and intercept.
    beta: A 2D Numpy array that has the beta values for each category.

    return: A 2D Numpy array that has the log-odds value for each observation,
    with the baseline column set to 0.
    """
    # Calculates log-odds and adds a column of 0's represent the baseline.
    odds = np.matmul(intercept, beta.T)
    odds = np.hstack([odds, np.zeros((intercept.shape[0], 1))])
    return odds


def softmax(odds):
    """
    This function implements the softmax coding, and turns the log-odds into
    probabilities.

    odds: Odds is a Numpy array that contains the log-odds values that are to
    be turned into probabilities.

    return: A 2D Numpy array that has the probability of each row.
    """
    exp_vals = np.exp(odds - np.max(odds, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


# print(softmax(odds_vals))


def ridge_likelihood(X, y, beta, tuning):
    """
    Finds the negative log-likelihood and implements ridge regression.

    X: A 2D Numpy array that contains the features for each observation.
    y: A 2D Numpy array that has the categories from the response variable.
    beta: A 2D Numpy array that is the coefficient matrix for the model.
    tuning: A float that is Regularization strength (λ)

    return: A float that is the calculated negative log-likelihood.
    """
    odds = log_odds(X, beta)
    prob = softmax(odds)

    # Takes the log of the probabilities from each category that corresponds
    # to the response variable and sums it up to get the negative log likelihood
    log_probs = np.log(prob[np.arange(1000), y])
    neg_likely = -np.sum(log_probs)

    # Adds penalty via ridge regression
    ridge_penalty = tuning * np.sum(beta[:, :-1] ** 2)

    return neg_likely + ridge_penalty


learning_rate = 0.00001
iterations = 1000
m = X_np.shape[0]


for i in range(iterations):
    y_pred = np.matmul(intercept_term, betas)
    error = y_pred - y
    gradients = (1/m) * (np.matmul(intercept_term.T, error))
    betas = betas - learning_rate * gradients

y_pred_gd = np.matmul(intercept_term, betas)





# reference industry: Education
# reference country: Australia
# reference severity: Critical


def main():
    # eda_hist()
    log_odds(X_np, betas)
    # print(odds_vals)

    print(ridge_likelihood(X, y, betas, 2.0))
    plt.show()


if __name__ == '__main__':
    main()
