
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_clusters = 5

def load_dataset(n_samples: int, n_clusters: int):
    """
    This function generates sample data with the given number of samples and clusters
    :param n_samples: number of samples to be generated
    :param n_clusters: number of clusters to be used in the generated data
    :return: generated data (X, y)
    """

    # Generate sample data
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=0)
    
    return X, y


# Load the dataset
X, y = load_dataset(600, n_clusters)

def kmeans_clustering(X, n_clusters):
    """
    This function performs k-means clustering on the given dataset with a specified number of clusters
    :param X: dataset on which k-means clustering should be performed
    :param n_clusters: number of clusters to be used in k-means clustering
    :return: k-means model object
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

# Perform k-means clustering
kmeans = kmeans_clustering(X, n_clusters)

def plot_kmeans_clusters(X, kmeans, x_label, y_label, title):
    """
    This function plots the k-means clustering results on a scatter plot
    :param X: dataset on which k-means clustering has been performed
    :param kmeans: k-means model object
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    """
    # Plot the data points with cluster assignments
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)

    # Plot the cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='D', color='black')

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=15)
    plt.legend(['Cluster', 'Cluster Center'])
    plt.show()

# Plot the k-means clustering results
plot_kmeans_clusters(X, kmeans, 'x-axis', 'y-axis', 'K-means Clustering')



# Comparing between different Countries
def read_worldbank_data(file_path: str):
    """
    This function reads in a file in Worldbank format and returns the original and transposed dataframe
    :param file_path: path of the file to be read
    :return: original dataframe and transposed dataframe
    """
    # read in the data
    original_df = pd.read_csv(file_path)
    
    # store the country names in a list
    country = list(original_df['Country Name'])
    
    # transpose the dataframe
    transposed_df = original_df.transpose()
    
    # update the column names with the country names
    transposed_df.columns = country
    
    # remove unnecessary rows
    transposed_df = transposed_df.iloc[4:]
    transposed_df = transposed_df.iloc[:-1]
    
    # reset index
    transposed_df = transposed_df.reset_index()
    
    # rename the index column
    transposed_df = transposed_df.rename(columns={"index": "Year"})
    
    # convert year column to int
    transposed_df['Year'] = transposed_df['Year'].astype(int)
    
    return original_df, transposed_df

df1, df2 = read_worldbank_data('data 2.csv')


def calculate_co2_stats(dataframe):
    """
    This function calculates the mean CO2 emissions (metric tons per capita) for a specific year
    :param dataframe: Dataframe containing the CO2 emissions data
    :return: None
    """
    # Calculate the mean CO2 emissions (metric tons per capita) for each country
    mean_ = dataframe.mean()
    print("Mean CO2 : ")
    print(mean_)
    print("\n\n")

    # Calculate the CO2 emissions (metric tons per capita) for each country in 2013
    _2013 = dataframe[dataframe['Year'] == 2013]
    print("CO2 2013 : ")
    print(_2013)

calculate_co2_stats(df2)

def plot_co2_over_time(df, countries):
    """
    This function plots the CO2 emissions (metric tons per capita) for a given list of countries over time
    :param df: dataframe containing CO2 emissions data
    :param countries: list of countries to plot CO2 emissions for
    """
    df.plot(x='Year', y=countries)
    plt.title("CO2 emissions for each country", fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("metric tons per capita", fontsize=14)
    plt.show()


country_name = list(df1['Country Name'])    
plot_co2_over_time(df2, country_name)




def read_worldbank_data(file_path: str):
    """
    This function reads in a file in Worldbank format and returns the original and transposed dataframe
    :param file_path: path of the file to be read
    :return: original dataframe and transposed dataframe
    """
    # read in the data
    original_df = pd.read_csv(file_path)
    
    # store the country names in a list
    country = list(original_df['Country Name'])
    
    # transpose the dataframe
    transposed_df = original_df.transpose()
    
    # update the column names with the country names
    transposed_df.columns = country
    
    # remove unnecessary rows
    transposed_df = transposed_df.iloc[4:]
    transposed_df = transposed_df.iloc[:-1]
    
    # reset index
    transposed_df = transposed_df.reset_index()
    
    # rename the index column
    transposed_df = transposed_df.rename(columns={"index": "Year"})
    
    # convert year column to int
    transposed_df['Year'] = transposed_df['Year'].astype(int)
    
    return original_df, transposed_df

original_df, transposed_df = read_worldbank_data("data 1.csv")

# store the country names in a list
country = list(original_df['Country Name'])


def plot_CO2_emissions(df, country):
    """
    This function plots the CO2 emissions (metric tons per capita) of a given country over time
    :param df: DataFrame containing the CO2 emissions data
    :param country: name of the country for which the CO2 emissions should be plotted
    """
    df.plot("Year", country, color="blue")
    plt.title(f"{country}'s CO2 emissions (metric tons per capita)", fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("metric tons per capita", fontsize=14)
    plt.legend(["CO2"])
    plt.show()

plot_CO2_emissions(transposed_df, country[0])




def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g
    :param t: input data
    :param n0: scale factor
    :param g: growth rate
    :param t0: inflection point
    :return: logistic function evaluated at t
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


df = transposed_df
param, covar = opt.curve_fit(logistic, df["Year"], df[country].squeeze(), p0=(float(df[country].iloc[0]), 0.03, 2000.0))

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

df["fit"] = logistic(df["Year"], *param)


def plot_logistic_fit(df, country):
    """
    Plot a logistic fit of the given country's CO2 emissions (metric tons per capita)
    :param df: DataFrame containing the CO2 data
    :param country: string of the country to plot
    """
    df.plot("Year", [country, "fit"], color=["blue", "red"])
    plt.title("Logistic fit of {}'s CO2 emissions".format(country), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("metric tons per capita", fontsize=14)
    plt.legend(["CO2"])
    plt.show()

plot_logistic_fit(df, country[0])


future_years = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
# generate logistic function values for future years
future_log = logistic(future_years, *param)
# unpack the parameters of the logistic function

# print the future logistic values
print("Future Logisitc Values 2021-2030 :")
print(future_log)


year = np.arange(df['Year'][0], 2031)
print(year)
forecast = logistic(year, *param)

def plot_future_prediction(df, country, forecast, year):
    """
    Plot the future prediction of a country's CO2 emissions (metric tons per capita) using a logistic model
    :param df: DataFrame containing the CO2 data
    :param country: string of the country to plot
    :param forecast: forecasted values for the future
    :param year: future year
    """
    plt.plot(df["Year"], df[country], label="CO2", color="blue")
    plt.plot(year, forecast, label="forecast", color="red")

    plt.title("Future Year prediction using Logistic of {}'s CO2 emissions".format(country), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("metric tons per capita", fontsize=14)
    plt.legend(["CO2"])
    plt.legend()
    plt.show()

plot_future_prediction(df, country[0], forecast, year)


df2 = pd.DataFrame({'Future Year': future_years, 'Logistic':future_log})
df2


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   



low, up = err_ranges(year, logistic, param, sigma)

def plot_prediction_limits(df, country, forecast, year, low, up):
    """
    Plot the future prediction of a country's CO2 emissions (metric tons per capita) using a logistic model along with lower and upper limits
    :param df: DataFrame containing the CO2 data
    :param country: string of the country to plot
    :param forecast: forecasted values for the future
    :param year: future year
    :param low: lower limits for the forecast
    :param up: upper limits for the forecast
    """
    plt.figure()
    plt.plot(df["Year"], df[country], label="CO2", color="blue")
    plt.plot(year, forecast, label="forecast", color="red")

    plt.fill_between(year, low, up, color="cyan", alpha=0.7)
    plt.title("Upper and Lower Limit of {}'s CO2 emissions".format(country), fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("metric tons per capita", fontsize=14)
    plt.legend()
    plt.show()

plot_prediction_limits(df, country[0], forecast, year, low, up)


print(err_ranges(2030, logistic, param, sigma))




