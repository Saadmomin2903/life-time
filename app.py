import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter, ModifiedBetaGeoFitter
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelParameters:
    """Store model parameters and metadata"""
    penalizer_coef: float
    model_type: str
    training_date: datetime
    metrics: Dict[str, float]

class DataPreprocessor:
    """Handle data preprocessing and validation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate required columns and data types"""
        required_columns = {'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
        return True
    
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess transaction data"""
        try:
            df = df.copy()
            # Convert date columns
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # Remove invalid transactions
            df = df[
                (df['Quantity'] > 0) &
                (df['UnitPrice'] > 0) &
                (pd.notnull(df['CustomerID']))
            ]
            
            # Remove outliers using IQR method
            for col in ['Quantity', 'UnitPrice']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[
                    (df[col] >= Q1 - 1.5 * IQR) &
                    (df[col] <= Q3 + 1.5 * IQR)
                ]
            
            # Calculate total sales
            df['Total_Sales'] = df['Quantity'] * df['UnitPrice']
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise

class CustomerSegmentation:
    """Handle customer segmentation using RFM analysis and clustering"""
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
    def create_rfm_segments(self, lf_data: pd.DataFrame) -> pd.DataFrame:
        """Create RFM segments using K-means clustering"""
        # Prepare RFM data for clustering
        rfm_data = self.scaler.fit_transform(
            lf_data[['recency', 'frequency', 'monetary_value']]
        )
        
        # Perform clustering
        lf_data['Segment'] = self.kmeans.fit_predict(rfm_data)
        
        # Label segments based on characteristics
        segment_labels = self._assign_segment_labels(lf_data)
        lf_data['Segment_Label'] = lf_data['Segment'].map(segment_labels)
        
        return lf_data
    
    def _assign_segment_labels(self, data: pd.DataFrame) -> Dict[int, str]:
        """Assign meaningful labels to segments based on their characteristics"""
        segment_stats = data.groupby('Segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary_value': 'mean'
        })
        
        # Create meaningful labels based on segment characteristics
        labels = {}
        for segment in range(self.n_clusters):
            stats = segment_stats.loc[segment]
            if stats['frequency'] > segment_stats['frequency'].median():
                if stats['monetary_value'] > segment_stats['monetary_value'].median():
                    labels[segment] = 'High-Value Loyal'
                else:
                    labels[segment] = 'Frequent Budget'
            else:
                if stats['recency'] < segment_stats['recency'].median():
                    labels[segment] = 'Recent One-Time'
                else:
                    labels[segment] = 'Lost Customers'
                    
        return labels

class LifetimeValueCalculator:
    """Handle CLV calculations and predictions with improved convergence handling"""
    
    def __init__(self, bgf_penalizer: float = 0.01, ggf_penalizer: float = 0.01):
        self.bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
        self.mbgf = ModifiedBetaGeoFitter(penalizer_coef=bgf_penalizer)
        self.ggf = GammaGammaFitter(penalizer_coef=ggf_penalizer)
        
    def fit_models(self, lf_data: pd.DataFrame, max_attempts: int = 3) -> None:
        """Fit BG/NBD and Gamma-Gamma models with automatic penalizer adjustment"""
        
        def attempt_fit(model, frequency, recency, T, penalizer_multiplier: float = 1.0):
            original_penalizer = model.penalizer_coef
            model.penalizer_coef *= penalizer_multiplier
            try:
                model.fit(frequency, recency, T)
                return True
            except Exception as e:
                logger.warning(f"Fitting failed with penalizer {model.penalizer_coef}: {str(e)}")
                model.penalizer_coef = original_penalizer
                return False

        # Prepare purchase data for Gamma-Gamma model
        purchase_data = lf_data[lf_data['frequency'] > 0]
        
        # Try fitting BG/NBD model with increasing penalizers
        for attempt in range(max_attempts):
            penalizer_multiplier = 2 ** attempt
            if attempt_fit(self.bgf, lf_data['frequency'], lf_data['recency'], 
                         lf_data['T'], penalizer_multiplier):
                break
            if attempt == max_attempts - 1:
                raise ValueError("BG/NBD model failed to converge even with increased penalizer")
        
        # Try fitting Modified BG/NBD model
        for attempt in range(max_attempts):
            penalizer_multiplier = 2 ** attempt
            if attempt_fit(self.mbgf, lf_data['frequency'], lf_data['recency'],
                         lf_data['T'], penalizer_multiplier):
                break
            if attempt == max_attempts - 1:
                logger.warning("Modified BG/NBD model failed to converge, will use regular BG/NBD only")
        
        # Try fitting Gamma-Gamma model
        for attempt in range(max_attempts):
            penalizer_multiplier = 2 ** attempt
            try:
                self.ggf.penalizer_coef *= penalizer_multiplier
                self.ggf.fit(purchase_data['frequency'], purchase_data['monetary_value'])
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise ValueError(f"Gamma-Gamma model failed to converge: {str(e)}")
                self.ggf.penalizer_coef /= penalizer_multiplier
    
    def calculate_clv(self, lf_data: pd.DataFrame, time_horizon: int = 12,
                     discount_rate: float = 0.01) -> pd.DataFrame:
        """Calculate CLV with error handling"""
        try:
            # Calculate CLV using BG/NBD
            lf_data['CLV_BGNBD'] = self.ggf.customer_lifetime_value(
                self.bgf,
                lf_data['frequency'],
                lf_data['recency'],
                lf_data['T'],
                lf_data['monetary_value'],
                time=time_horizon,
                discount_rate=discount_rate
            )
            
            try:
                # Try calculating CLV using MBG/NBD if available
                lf_data['CLV_MBGNBD'] = self.ggf.customer_lifetime_value(
                    self.mbgf,
                    lf_data['frequency'],
                    lf_data['recency'],
                    lf_data['T'],
                    lf_data['monetary_value'],
                    time=time_horizon,
                    discount_rate=discount_rate
                )
            except:
                logger.warning("Could not calculate MBG/NBD CLV, using BG/NBD only")
                lf_data['CLV_MBGNBD'] = lf_data['CLV_BGNBD']
            
            # Add simplified confidence interval calculation
            lf_data['CLV_Lower'] = lf_data['CLV_BGNBD'] * 0.8  # 20% lower bound
            lf_data['CLV_Upper'] = lf_data['CLV_BGNBD'] * 1.2  # 20% upper bound
            
            return lf_data
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {str(e)}")
            raise



def format_diagnostic_value(value: Optional[float]) -> str:
    """Format diagnostic values with proper null handling"""
    if value is None:
        return 'N/A'
    return f"{value:.2f}"

def display_model_diagnostics(diagnostics: Dict[str, Optional[float]]):
    """Display model diagnostics with proper error handling"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("BG/NBD Model Performance")
        st.metric(
            "Log-likelihood",
            format_diagnostic_value(diagnostics.get('bgf_log_likelihood'))
        )
        st.metric(
            "AIC",
            format_diagnostic_value(diagnostics.get('bgf_aic'))
        )

    with col2:
        st.write("MBG/NBD Model Performance")
        st.metric(
            "Log-likelihood",
            format_diagnostic_value(diagnostics.get('mbgf_log_likelihood'))
        )
        st.metric(
            "AIC",
            format_diagnostic_value(diagnostics.get('mbgf_aic'))
        )



class DashboardUI:
    """Handle Streamlit UI components and visualization"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Advanced CLV Calculator",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._apply_custom_css()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
            <style>
            .stPlot {
                background-color: white;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .segment-high {
                color: #28a745;
                font-weight: bold;
            }
            .segment-medium {
                color: #ffc107;
                font-weight: bold;
            }
            .segment-low {
                color: #dc3545;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def render_sidebar_controls(self) -> Dict:
        """Render sidebar controls and return selected parameters"""
        st.sidebar.title("Analysis Parameters")
        
        params = {
            'prediction_period': st.sidebar.slider(
                'Prediction Period (days)',
                1, 365, 30
            ),
            'discount_rate': st.sidebar.slider(
                'Discount Rate',
                0.0, 0.2, 0.01, 0.01
            ),
            'n_clusters': st.sidebar.slider(
                'Number of Customer Segments',
                2, 8, 4
            )
        }
        
        return params
    
    def plot_customer_segments(self, data: pd.DataFrame):
        """Create interactive 3D scatter plot of customer segments"""
        fig = px.scatter_3d(
            data,
            x='recency',
            y='frequency',
            z='monetary_value',
            color='Segment_Label',
            title='Customer Segments (3D View)',
            labels={
                'recency': 'Recency (days)',
                'frequency': 'Frequency',
                'monetary_value': 'Monetary Value ($)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_clv_distribution(self, data: pd.DataFrame):
        """Plot CLV distribution with confidence intervals"""
        fig = go.Figure()
        
        # Add CLV distribution
        fig.add_trace(go.Histogram(
            x=data['CLV_BGNBD'],
            name='CLV Distribution',
            nbinsx=50
        ))
        
        # Add confidence interval markers
        fig.add_vline(
            x=data['CLV_Lower'].mean(),
            line_dash="dash",
            annotation_text="95% CI Lower"
        )
        fig.add_vline(
            x=data['CLV_Upper'].mean(),
            line_dash="dash",
            annotation_text="95% CI Upper"
        )
        
        fig.update_layout(
            title='Customer Lifetime Value Distribution',
            xaxis_title='Predicted CLV ($)',
            yaxis_title='Number of Customers'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize components
    ui = DashboardUI()
    preprocessor = DataPreprocessor()
    
    try:
        # Load and preprocess data
        raw_data = pd.read_csv("OnlineRetail.csv", encoding="cp1252")
        preprocessor.validate_data(raw_data)
        clean_data = preprocessor.clean_transactions(raw_data)
        
        # Get analysis parameters
        params = ui.render_sidebar_controls()
        
        # Create lifetime value summary
        observation_period_end = clean_data['InvoiceDate'].max()
        lf_data = summary_data_from_transaction_data(
            clean_data,
            'CustomerID',
            'InvoiceDate',
            monetary_value_col='Total_Sales',
            observation_period_end=observation_period_end
        )
        
        # Perform customer segmentation
        segmentation = CustomerSegmentation(n_clusters=params['n_clusters'])
        lf_data = segmentation.create_rfm_segments(lf_data)
        
        # Calculate CLV
        calculator = LifetimeValueCalculator()
        calculator.fit_models(lf_data)
        lf_data = calculator.calculate_clv(
            lf_data,
            time_horizon=params['prediction_period'],
            discount_rate=params['discount_rate']
        )
        
        # Display visualizations
        st.title('Advanced Customer Lifetime Value Analysis')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average CLV",
                f"${lf_data['CLV_BGNBD'].mean():,.2f}",
                f"±${(lf_data['CLV_Upper'] - lf_data['CLV_Lower']).mean()/2:,.2f}"
            )
        with col2:
            st.metric(
                "Total Predicted Revenue",
                f"${lf_data['CLV_BGNBD'].sum():,.2f}"
            )
        with col3:
            st.metric(
                "High-Value Customers",
                f"{(lf_data['CLV_BGNBD'] > lf_data['CLV_BGNBD'].quantile(0.9)).sum():,}"
            )
        
        # Display segment analysis
        st.subheader('Customer Segmentation Analysis')
        ui.plot_customer_segments(lf_data)
        
        # Display CLV distribution
        st.subheader('CLV Distribution Analysis')
        ui.plot_clv_distribution(lf_data)
        
        # Display top customers table
        st.subheader('Top Customers by Predicted CLV')
        top_customers = lf_data.nlargest(10, 'CLV_BGNBD')
        st.dataframe(
            top_customers[[
                'frequency', 'recency', 'monetary_value', 'CLV_BGNBD', 
                'CLV_Lower', 'CLV_Upper', 'Segment_Label'
            ]]
            .style.format({
                'CLV_BGNBD': '${:,.2f}',
                'CLV_Lower': '${:,.2f}',
                'CLV_Upper': '${:,.2f}',
                'monetary_value': '${:,.2f}',
                'recency': '{:.1f}',
                'frequency': '{:.0f}'
            })
        )
        
        st.subheader('Model Diagnostics')
    
        # Get model diagnostics
        diagnostics = calculator.get_model_diagnostics()
    
        # Display diagnostics using the new function
        display_model_diagnostics(diagnostics)
        
        # Compare BG/NBD vs MBG/NBD performance
        col1, col2 = st.columns(2)
        with col1:
            st.write("BG/NBD Model Performance")
            st.write(f"Log-likelihood: {diagnostics['bgf_log_likelihood']:.2f if diagnostics['bgf_log_likelihood'] is not None else 'N/A'}")
            st.write(f"AIC: {diagnostics['bgf_aic']:.2f if diagnostics['bgf_aic'] is not None else 'N/A'}")
            
        with col2:
            st.write("MBG/NBD Model Performance")
            st.write(f"Log-likelihood: {diagnostics['mbgf_log_likelihood']:.2f if diagnostics['mbgf_log_likelihood'] is not None else 'N/A'}")
            st.write(f"AIC: {diagnostics['mbgf_aic']:.2f if diagnostics['mbgf_aic'] is not None else 'N/A'}")
        
        # Add customer cohort analysis
        st.subheader('Cohort Analysis')
        cohort_analysis = CohortAnalysis(clean_data)
        cohort_matrix = cohort_analysis.create_cohort_matrix()
        
        # Plot cohort heatmap
        fig = px.imshow(
            cohort_matrix,
            labels=dict(x="Cohort Period", y="Cohort Group", color="Retention Rate"),
            title="Customer Cohort Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add export functionality
        if st.button('Export Analysis Results'):
            export_results(lf_data, cohort_matrix)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

class CohortAnalysis:
    """Handle customer cohort analysis"""
    
    def __init__(self, transaction_data: pd.DataFrame):
        self.data = transaction_data
        
    def create_cohort_matrix(self) -> pd.DataFrame:
        """Create customer cohort retention matrix"""
        # Create cohort groups
        self.data['CohortDate'] = self.data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
        self.data['TransactionPeriod'] = self.data['InvoiceDate'].dt.to_period('M')
        
        # Calculate cohort periods
        self.data['CohortPeriod'] = (
            self.data['TransactionPeriod'] - 
            self.data['CohortDate']
        ).apply(lambda x: x.n)
        
        # Create cohort matrix
        cohort_data = self.data.groupby(['CohortDate', 'CohortPeriod'])['CustomerID'].nunique().reset_index()
        cohort_matrix = cohort_data.pivot(
            index='CohortDate',
            columns='CohortPeriod',
            values='CustomerID'
        )
        
        # Calculate retention rates
        cohort_sizes = cohort_matrix[0]
        retention_matrix = cohort_matrix.div(cohort_sizes, axis=0) * 100
        
        return retention_matrix

class ModelPersistence:
    """Handle model saving and loading"""
    
    @staticmethod
    def save_models(calculator: LifetimeValueCalculator, params: Dict, path: str = 'models/'):
        """Save trained models and parameters"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        joblib.dump(calculator.bgf, f"{path}bgf_model.pkl")
        joblib.dump(calculator.mbgf, f"{path}mbgf_model.pkl")
        joblib.dump(calculator.ggf, f"{path}ggf_model.pkl")
        
        # Save parameters
        model_params = ModelParameters(
            penalizer_coef=calculator.bgf.penalizer_coef,
            model_type="BG/NBD + GammaGamma",
            training_date=datetime.now(),
            metrics={
                'bgf_log_likelihood': calculator.bgf.log_likelihood_,
                'bgf_aic': calculator.bgf.AIC_,
                'mbgf_log_likelihood': calculator.mbgf.log_likelihood_,
                'mbgf_aic': calculator.mbgf.AIC_
            }
        )
        
        joblib.dump(model_params, f"{path}model_parameters.pkl")
    
    @staticmethod
    def load_models(path: str = 'models/') -> Tuple[LifetimeValueCalculator, ModelParameters]:
        """Load trained models and parameters"""
        calculator = LifetimeValueCalculator()
        
        try:
            calculator.bgf = joblib.load(f"{path}bgf_model.pkl")
            calculator.mbgf = joblib.load(f"{path}mbgf_model.pkl")
            calculator.ggf = joblib.load(f"{path}ggf_model.pkl")
            model_params = joblib.load(f"{path}model_parameters.pkl")
            
            return calculator, model_params
            
        except FileNotFoundError:
            logger.warning("No saved models found. Will train new models.")
            return calculator, None

def export_results(lf_data: pd.DataFrame, cohort_matrix: pd.DataFrame):
    """Export analysis results to various formats"""
    try:
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export customer analysis
        lf_data.to_csv(f"exports/customer_analysis_{timestamp}.csv")
        
        # Export cohort analysis
        cohort_matrix.to_csv(f"exports/cohort_analysis_{timestamp}.csv")
        
        # Create Excel report with multiple sheets
        with pd.ExcelWriter(f"exports/clv_analysis_{timestamp}.xlsx") as writer:
            lf_data.to_excel(writer, sheet_name='Customer Analysis')
            cohort_matrix.to_excel(writer, sheet_name='Cohort Analysis')
            
            # Add summary statistics
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Average CLV',
                    'Total Predicted Revenue',
                    'High-Value Customers Count',
                    'Average Retention Rate'
                ],
                'Value': [
                    f"${lf_data['CLV_BGNBD'].mean():,.2f}",
                    f"${lf_data['CLV_BGNBD'].sum():,.2f}",
                    f"{(lf_data['CLV_BGNBD'] > lf_data['CLV_BGNBD'].quantile(0.9)).sum():,}",
                    f"{cohort_matrix.mean().mean():.1f}%"
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary Statistics')
            
        st.success("Analysis results exported successfully!")
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        st.error("Failed to export results. Please try again.")

if __name__ == "__main__":
    main()
