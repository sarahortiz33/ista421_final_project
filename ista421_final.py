import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


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


def eda_hist(country_freq, industry_freq, severe_freq):
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


def ridge_likelihood(X, y, beta, tuning):
    """
    Finds the negative log-likelihood and implements ridge regression.

    X: A 2D Numpy array that contains the features for each observation.
    y: A Numpy array that has the categories from the response variable.
    beta: A 2D Numpy array that is the coefficient matrix for the model.
    tuning: A float that is Regularization strength (λ)

    return: A float that is the calculated negative log-likelihood.
    """
    odds = log_odds(X, beta)
    prob = softmax(odds)

    # Takes the log of the probabilities from each category that corresponds
    # to the response variable and sums it up to get the negative log likelihood
    log_probs = np.log(prob[np.arange(X.shape[0]), y])
    neg_likely = -np.sum(log_probs)

    # Adds penalty via ridge regression
    ridge_penalty = tuning * np.sum(beta[:, :-1] ** 2)

    return neg_likely + ridge_penalty


def gradient_descent(X, y, beta, learning_rate, iterations, tuning):
    n_category = beta.shape[0]
    loss = {}

    # Loops through to calculate the gradient and update the beta values.
    for i in range(iterations):
        odds = log_odds(X, beta)
        prob = softmax(odds)

        # predicted p-y
        prob[np.arange(X.shape[0]), y] -= 1

        # Multiplication with feature variables
        gradients = np.matmul(prob[:, :n_category].T, X) / X.shape[0]
        gradients += tuning * beta

        beta = beta - learning_rate * gradients

        if i % 50 == 0:
            loss[i] = float(ridge_likelihood(X, y, beta, tuning))

    return beta, loss


def loss_plot(loss):
    pass


def k_cv(X, y, n_severity):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[val_index]
        y_train = y[train_index]
        y_test = y[val_index]

        fold = np.zeros((n_severity, X_train.shape[1]))

        grad_des = gradient_descent(X_train, y_train, fold,
                                    learning_rate=0.01, iterations=1000,
                                    tuning=0.1)
        fold = grad_des[0]

        val_odds = log_odds(X_test, fold)
        val_probs = softmax(val_odds)





def main():
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

    country_freq = find_freq(country)
    industry_freq = find_freq(industry)
    severe_freq = find_freq(severity)

    #eda_hist(country_freq, industry_freq, severe_freq)

    # One hot-encoded variables
    df_dumb = pd.get_dummies(df, columns=['country', 'industry'], drop_first=True, dtype=int)
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

    n_classes = 4
    n_severity = n_classes - 1

    # Set betas to 0
    betas = np.zeros((n_severity, X_np.shape[1]))

    # !!!!RANDOM NUMBER BETA!!!!!
    # np.random.seed(42)
    # betas = np.random.randn(n_severity, X_np.shape[1])
    # print(betas)

    # Begin to compute gradient descent
    grad_des = gradient_descent(X_np, y, betas, 0.01, 1000, 0.1)
    new_betas = grad_des[0]
    loss = grad_des[1]

    print(loss)

    k_cv(X_np, y, n_severity)

    plt.show()


if __name__ == '__main__':
    main()
