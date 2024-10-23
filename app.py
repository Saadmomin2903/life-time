import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Lifetime Value Calculator",
    layout="wide"
)

# Custom CSS to improve app appearance
st.markdown("""
    <style>
    .stPlot {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess transaction data."""
    try:
        tx_data = pd.read_csv("OnlineRetail.csv", encoding="cp1252")
        
        # Data cleaning and preprocessing
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
        tx_data = tx_data[pd.notnull(tx_data['CustomerID'])]
        tx_data = tx_data[tx_data['Quantity'] > 0]
        tx_data['Total_Sales'] = tx_data['Quantity'] * tx_data['UnitPrice']
        
        return tx_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_summary_metrics(lf_tx_data):
    """Create summary metrics for the dashboard."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{lf_tx_data.index.nunique():,}")
    with col2:
        avg_frequency = lf_tx_data['frequency'].mean()
        st.metric("Average Purchase Frequency", f"{avg_frequency:.2f}")
    with col3:
        avg_monetary = lf_tx_data['monetary_value'].mean()
        st.metric("Average Order Value", f"${avg_monetary:.2f}")

def plot_frequency_histogram(lf_tx_data):
    """Create purchase frequency histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=lf_tx_data, x='frequency', bins=50, ax=ax)
    ax.set_title('Purchase Frequency Distribution')
    ax.set_xlabel('Number of Repeat Purchases')
    ax.set_ylabel('Count of Customers')
    st.pyplot(fig)
    plt.close()

def train_models(lf_tx_data):
    """Train BG/NBD and Gamma-Gamma models."""
    # BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'])
    
    # Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0)
    shortlisted_customers = lf_tx_data[lf_tx_data['frequency'] > 0]
    ggf.fit(shortlisted_customers['frequency'], shortlisted_customers['monetary_value'])
    
    return bgf, ggf

def calculate_clv(bgf, ggf, lf_tx_data):
    """Calculate Customer Lifetime Value."""
    lf_tx_data['CLV'] = ggf.customer_lifetime_value(
        bgf,
        lf_tx_data['frequency'],
        lf_tx_data['recency'],
        lf_tx_data['T'],
        lf_tx_data['monetary_value'],
        time=12,
        discount_rate=0.01
    )
    return lf_tx_data

def main():
    st.title('Customer Lifetime Value Calculator')
    
    # Load data
    tx_data = load_data()
    
    if tx_data is not None:
        # Calculate observation period end
        observation_period_end = tx_data['InvoiceDate'].max()
        
        # Create lifetime value summary
        lf_tx_data = summary_data_from_transaction_data(
            tx_data,
            'CustomerID',
            'InvoiceDate',
            monetary_value_col='Total_Sales',
            observation_period_end=observation_period_end
        )
        
        # Display summary metrics
        create_summary_metrics(lf_tx_data)
        
        # Display frequency histogram
        st.subheader('Purchase Patterns Analysis')
        plot_frequency_histogram(lf_tx_data)
        
        # Calculate and display one-time buyers percentage
        one_time_buyers = round(sum(lf_tx_data['frequency'] == 0) / len(lf_tx_data) * 100, 2)
        st.info(f"📊 {one_time_buyers}% of customers are one-time buyers")
        
        # Train models
        with st.spinner('Training models...'):
            bgf, ggf = train_models(lf_tx_data)
        
        # Predict future transactions
        st.subheader('Future Transaction Predictions')
        prediction_period = st.slider('Select prediction period (days)', 1, 365, 30)
        
        lf_tx_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            prediction_period,
            lf_tx_data['frequency'],
            lf_tx_data['recency'],
            lf_tx_data['T']
        )
        
        # Calculate and display CLV
        lf_tx_data = calculate_clv(bgf, ggf, lf_tx_data)
        
        # Display top customers
        st.subheader('Top Customers by Predicted CLV')
        top_customers = lf_tx_data.sort_values('CLV', ascending=False).head(10)
        
        st.dataframe(
            top_customers[['frequency', 'recency', 'T', 'monetary_value', 'predicted_purchases', 'CLV']]
            .style.format({
                'CLV': '${:,.2f}',
                'monetary_value': '${:,.2f}',
                'predicted_purchases': '{:,.1f}'
            })
        )

if __name__ == "__main__":
    main()
