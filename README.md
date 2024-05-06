


#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
Here, we're importing two libraries: pandas for data manipulation and matplotlib.pyplot for plotting graphs.


#Import Data
tx_data=pd.read_csv("OnlineRetail.csv", encoding="cp1252")
tx_data.head()
This reads a CSV file named "OnlineRetail.csv" into a pandas DataFrame called tx_data and displays the first few rows of the DataFrame.


#Check the shape (number of columns and rows) in the dataset
tx_data.shape
This line prints the number of rows and columns in the dataset.

#Find out missing values
tx_data.isnull().sum(axis=0)
This checks for missing values in each column of the dataset and prints the total count of missing values for each column.


#Remove time from date
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'], format="%m/%d/%Y %H:%M").dt.date
This line converts the 'InvoiceDate' column to datetime format and keeps only the date part, removing the time.

#There are 135,080 missing values in the CustomerID column, and since our analysis is based on customers, 
#we will remove these missing values.
tx_data = tx_data[pd.notnull(tx_data['CustomerID'])]
This removes rows with missing values in the 'CustomerID' column since the analysis is focused on customers.


#Keep records with non-negative quantity
tx_data = tx_data[(tx_data['Quantity']>0)]
This keeps only the records where the 'Quantity' column has values greater than 0, ensuring we only have valid transactions.


#Add a new column depicting total sales
tx_data['Total_Sales'] = tx_data['Quantity'] * tx_data['UnitPrice']
This line calculates the total sales by multiplying the quantity purchased by the unit price of each item and adds a new column 'Total_Sales' to the DataFrame.


necessary_cols = ['CustomerID', 'InvoiceDate', 'Total_Sales']
tx_data = tx_data[necessary_cols]
tx_data.head()
This selects only the necessary columns ('CustomerID', 'InvoiceDate', 'Total_Sales') and updates the DataFrame with these columns. Then it displays the first few rows of the updated DataFrame.

This summarizes the data preparation steps where we import the data, handle missing values, transform the data, and select necessary columns for further analysis.




Copy code
print(tx_data['CustomerID'].nunique())
This line prints the number of unique customer IDs present in the dataset.


last_order_date = tx_data['InvoiceDate'].max()
print(last_order_date)
Here, we find the latest (maximum) order date from the 'InvoiceDate' column and print it out.


print("--------------------------------------")
print(tx_data[(tx_data['CustomerID']==12346)])
These lines print a separator followed by details of transactions made by a specific customer with the ID '12346'.


from lifetimes.plotting import *
from lifetimes.utils import *
These lines import necessary functions from the lifetimes package for plotting and utility purposes.


lf_tx_data = summary_data_from_transaction_data(tx_data, 'CustomerID', 'InvoiceDate', monetary_value_col='Total_Sales', observation_period_end='2011-12-9')
lf_tx_data.reset_index().head()
Here, we summarize transactional data into a suitable format for customer lifetime value analysis using the 'summary_data_from_transaction_data' function from the lifetimes package. Then, we reset the index of the resulting DataFrame and display the first few rows.


%matplotlib inline
This magic command is used in Jupyter notebooks to display matplotlib plots inline within the notebook.


lf_tx_data['frequency'].plot(kind='hist', bins=50)
This line creates a histogram to visualize the frequency distribution of customer transactions.


print(lf_tx_data['frequency'].describe())
Here, we print descriptive statistics (like mean, median, min, max) for the 'frequency' column, which represents how often customers make purchases.


print("---------------------------------------")
one_time_buyers = round(sum(lf_tx_data['frequency'] == 0)/float(len(lf_tx_data))*(100),2)
print("Percentage of customers purchase the item only once:", one_time_buyers ,"%")
These lines calculate and print the percentage of customers who made only one purchase (one-time buyers).

This concludes the frequency analysis section where we examine how often customers make purchases and the percentage of one-time buyers.



from lifetimes import BetaGeoFitter
This line imports the BetaGeoFitter class from the lifetimes package, which is used to fit the Beta-Geometric/Negative Binomial Distribution (BG/NBD) model for customer lifetime value analysis.


bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'])
Here, we initialize an instance of the BetaGeoFitter class with a penalizer coefficient of 0.0, which avoids overfitting. Then, we fit the BG/NBD model to the summarized transactional data using the 'fit' method.


print(bgf)
This prints out the fitted BG/NBD model object with its parameters.


bgf.summary
This prints a summary of the BG/NBD model, including coefficients, standard errors, and confidence intervals.


coef	se(coef)	lower 95% bound	upper 95% bound
r	0.826433	0.026780	0.773944	0.878922
alpha	68.890678	2.611055	63.773011	74.008345
a	0.003443	0.010347	-0.016837	0.023722
b	6.749363	22.412933	-37.179985	50.678711
This output displays the coefficients, standard errors, and 95% confidence intervals for each parameter in the fitted BG/NBD model:

r: Represents the probability of a customer being alive (i.e., still making transactions).
alpha: Represents the shape parameter of the gamma distribution for the transaction process.
a: Represents the shape parameter of the gamma distribution for the dropout process.
b: Represents the scale parameter of the gamma distribution for the dropout process.
These values provide insights into the underlying distribution of customer behavior captured by the BG/NBD model.


Copy code
%matplotlib inline
This magic command is used in Jupyter notebooks to display matplotlib plots inline within the notebook.


import matplotlib.pyplot as plt
Here, we import the pyplot module from the matplotlib library, which provides functions to create various types of plots and visualizations.


from lifetimes.plotting import plot_frequency_recency_matrix
This line imports the function 'plot_frequency_recency_matrix' from the lifetimes.plotting module, which is used to visualize the frequency-recency matrix of customers.


fig = plt.figure(figsize=(12,8))
plot_frequency_recency_matrix(bgf)
These lines create a figure with a specified size and then plot the frequency-recency matrix using the fitted BG/NBD model.


from lifetimes.plotting import plot_probability_alive_matrix
This imports the function 'plot_probability_alive_matrix' from the lifetimes.plotting module, which is used to visualize the probability of a customer being alive.


fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)
These lines create a figure with a specified size and then plot the probability of a customer being alive matrix using the fitted BG/NBD model.


t = 10
lf_tx_data['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T']),2)
This line predicts the number of transactions each customer is expected to make in the next 10 days using the fitted BG/NBD model and stores the predicted values in a new column called 'pred_num_txn' in the DataFrame.


lf_tx_data.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()
This line sorts the DataFrame by the predicted number of transactions ('pred_num_txn') in descending order and displays the top 10 customers who are expected to make the most purchases in the next 10 days.





from lifetimes.plotting import plot_period_transactions
This line imports the function 'plot_period_transactions' from the lifetimes.plotting module, which is used to visualize the actual versus predicted number of transactions over time.


plot_period_transactions(bgf)
This line plots the actual versus predicted number of transactions over time using the fitted BG/NBD model.


t = 10
individual = lf_tx_data.loc[14911]
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
These lines predict the number of transactions for a specific customer (with CustomerID 14911) in the next 10 days using the fitted BG/NBD model.


lf_tx_data[['monetary_value', 'frequency']].corr()
This line calculates the correlation between the 'monetary_value' (average transaction value) and 'frequency' (number of transactions) columns in the DataFrame.


shortlisted_customers = lf_tx_data[lf_tx_data['frequency']>0]
This line filters the DataFrame to include only customers who have made more than one purchase, i.e., customers with a frequency greater than 0.


print(shortlisted_customers.head().reset_index())
This line prints the first few rows of the DataFrame containing shortlisted customers who have made repeat purchases.

print("-----------------------------------------")
print("The Number of Returning Customers are: ",len(shortlisted_customers))
These lines print a separator followed by the count of returning customers, i.e., customers who have made repeat purchases.




