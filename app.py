import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Load data
@st.cache
def load_data():
    try:
        tx_data = pd.read_csv("OnlineRetail.csv", encoding="cp1252")
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'], format="%m/%d/%Y %H:%M").dt.date
        tx_data = tx_data[pd.notnull(tx_data['CustomerID'])]
        tx_data = tx_data[(tx_data['Quantity'] > 0)]
        tx_data['Total_Sales'] = tx_data['Quantity'] * tx_data['UnitPrice']
        return tx_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

tx_data = load_data()

if tx_data is not None:
    # Summary data
    lf_tx_data = summary_data_from_transaction_data(tx_data, 'CustomerID', 'InvoiceDate', monetary_value_col='Total_Sales', observation_period_end='2011-12-9')

    # Build Streamlit app
    st.title('Customer Lifetime Value Calculator')

    # Display dataset summary
    st.subheader('Dataset Summary')
    st.write(f"Number of unique customers: {lf_tx_data.index.nunique()}")

    # Histogram of purchase frequencies
    st.subheader('Purchase Frequency Histogram')
    plt.hist(lf_tx_data['frequency'], bins=50)
    st.pyplot(plt)

    # Percentage of one-time buyers
    one_time_buyers = round(sum(lf_tx_data['frequency'] == 0) / len(lf_tx_data) * 100, 2)
    st.write(f"Percentage of customers purchase the item only once: {one_time_buyers}%")

    # Frequency/Recency Analysis Using the BG/NBD Model
    st.subheader('Frequency/Recency Analysis Using the BG/NBD Model')
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'])
    st.write(bgf.summary)

    # Predict future transactions
    st.subheader('Predict Future Transactions')
    t = 10
    lf_tx_data['pred_num_txn'] = round(
        bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_tx_data['frequency'], lf_tx_data['recency'],
                                                                lf_tx_data['T']), 2)
    st.write(lf_tx_data.sort_values(by='pred_num_txn', ascending=False).head(10))

    # Customer's future transaction prediction for next 10 days
    st.subheader("Customer's Future Transaction Prediction for Next 10 Days")
    individual = lf_tx_data.loc[14911]
    predicted_transactions = bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
    st.write(f"Predicted transactions for customer 14911 in the next {t} days: {predicted_transactions}")

    # Train Gamma-Gamma model
    st.subheader('Train Gamma-Gamma Model')
    shortlisted_customers = lf_tx_data[lf_tx_data['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(shortlisted_customers['frequency'], shortlisted_customers['monetary_value'])
    st.write(ggf.summary)

    # Calculate Customer Lifetime Value
    st.subheader('Calculate Customer Lifetime Value')
    lf_tx_data['CLV'] = round(
        ggf.customer_lifetime_value(bgf, lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'],
                                    lf_tx_data['monetary_value'], time=12, discount_rate=0.01), 2)
    st.write(lf_tx_data.sort_values(by='CLV', ascending=False).head(10))
