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
    """Handle CLV calculations and predictions with adaptive penalization"""
    
    def __init__(self, initial_penalizer: float = 0.001, max_penalizer: float = 10.0):
        self.initial_penalizer = initial_penalizer
        self.max_penalizer = max_penalizer
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models with current penalizer value"""
        self.bgf = BetaGeoFitter(penalizer_coef=self.initial_penalizer)
        self.mbgf = ModifiedBetaGeoFitter(penalizer_coef=self.initial_penalizer)
        self.ggf = GammaGammaFitter(penalizer_coef=self.initial_penalizer)
    
    def _fit_with_adaptive_penalization(self, model, frequency, recency, T, 
                                      monetary_value=None) -> Tuple[bool, str]:
        """
        Attempt to fit model with increasingly larger penalizers until convergence
        """
        current_penalizer = self.initial_penalizer
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                if isinstance(model, GammaGammaFitter) and monetary_value is not None:
                    model.fit(frequency, monetary_value)
                else:
                    model.fit(frequency, recency, T)
                return True, "Model converged successfully"
                
            except Exception as e:
                if "did not converge" in str(e) and current_penalizer < self.max_penalizer:
                    # Increase penalizer geometrically
                    current_penalizer *= 2
                    logger.warning(f"Model didn't converge. Increasing penalizer to {current_penalizer}")
                    
                    # Reinitialize model with new penalizer
                    if isinstance(model, BetaGeoFitter):
                        model = BetaGeoFitter(penalizer_coef=current_penalizer)
                    elif isinstance(model, ModifiedBetaGeoFitter):
                        model = ModifiedBetaGeoFitter(penalizer_coef=current_penalizer)
                    elif isinstance(model, GammaGammaFitter):
                        model = GammaGammaFitter(penalizer_coef=current_penalizer)
                else:
                    return False, f"Failed to converge after {attempt + 1} attempts. Last error: {str(e)}"
        
        return False, f"Failed to converge after {max_attempts} attempts with max penalizer {current_penalizer}"
    
    def _preprocess_data(self, lf_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data to improve model convergence"""
        # Remove extreme outliers
        for col in ['frequency', 'recency', 'T', 'monetary_value']:
            Q1 = lf_data[col].quantile(0.01)
            Q3 = lf_data[col].quantile(0.99)
            IQR = Q3 - Q1
            lf_data = lf_data[
                (lf_data[col] >= Q1 - 1.5 * IQR) &
                (lf_data[col] <= Q3 + 1.5 * IQR)
            ]
        
        # Scale monetary values to prevent numerical issues
        if 'monetary_value' in lf_data.columns:
            lf_data['monetary_value'] = lf_data['monetary_value'] / lf_data['monetary_value'].max()
        
        return lf_data
    
    def fit_models(self, lf_data: pd.DataFrame) -> Dict[str, Tuple[bool, str]]:
        """Fit BG/NBD and Gamma-Gamma models with convergence handling"""
        # Preprocess data
        lf_data = self._preprocess_data(lf_data)
        
        # Initialize status dictionary
        fitting_status = {}
        
        # Fit BG/NBD model
        bgf_status = self._fit_with_adaptive_penalization(
            self.bgf,
            lf_data['frequency'],
            lf_data['recency'],
            lf_data['T']
        )
        fitting_status['bgf'] = bgf_status
        
        # Fit Modified BG/NBD model
        mbgf_status = self._fit_with_adaptive_penalization(
            self.mbgf,
            lf_data['frequency'],
            lf_data['recency'],
            lf_data['T']
        )
        fitting_status['mbgf'] = mbgf_status
        
        # Fit Gamma-Gamma model only for customers with purchases
        purchase_data = lf_data[lf_data['frequency'] > 0].copy()
        if len(purchase_data) > 0:
            ggf_status = self._fit_with_adaptive_penalization(
                self.ggf,
                purchase_data['frequency'],
                None,
                None,
                purchase_data['monetary_value']
            )
            fitting_status['ggf'] = ggf_status
        else:
            fitting_status['ggf'] = (False, "No customers with purchases found")
        
        return fitting_status

    def calculate_clv(self, lf_data: pd.DataFrame, time_horizon: int = 12,
                     discount_rate: float = 0.01) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculate CLV using available converged models"""
        results = lf_data.copy()
        model_metrics = {}
        
        try:
            # Calculate CLV using BG/NBD if available
            if hasattr(self.bgf, 'params_'):
                results['CLV_BGNBD'] = self.ggf.customer_lifetime_value(
                    self.bgf,
                    results['frequency'],
                    results['recency'],
                    results['T'],
                    results['monetary_value'],
                    time=time_horizon,
                    discount_rate=discount_rate
                )
                model_metrics['bgf_fit'] = {
                    'log_likelihood': self.bgf.log_likelihood_,
                    'AIC': self.bgf.AIC_,
                    'penalizer': self.bgf.penalizer_coef
                }
            
            # Calculate CLV using MBG/NBD if available
            if hasattr(self.mbgf, 'params_'):
                results['CLV_MBGNBD'] = self.ggf.customer_lifetime_value(
                    self.mbgf,
                    results['frequency'],
                    results['recency'],
                    results['T'],
                    results['monetary_value'],
                    time=time_horizon,
                    discount_rate=discount_rate
                )
                model_metrics['mbgf_fit'] = {
                    'log_likelihood': self.mbgf.log_likelihood_,
                    'AIC': self.mbgf.AIC_,
                    'penalizer': self.mbgf.penalizer_coef
                }
            
            # Calculate confidence intervals if both models converged
            if hasattr(self.bgf, 'params_') and hasattr(self.ggf, 'params_'):
                results['CLV_Lower'], results['CLV_Upper'] = self._calculate_clv_confidence_intervals(
                    results, time_horizon, discount_rate
                )
            
        except Exception as e:
            logger.error(f"Error in CLV calculation: {str(e)}")
            raise ValueError(f"CLV calculation failed: {str(e)}")
        
        return results, model_metrics

def main():
    try:
        # Initialize components
        ui = DashboardUI()
        preprocessor = DataPreprocessor()
        
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
        
        # Initialize calculator with conservative initial penalizer
        calculator = LifetimeValueCalculator(initial_penalizer=0.001)
        
        # Fit models and get status
        fitting_status = calculator.fit_models(lf_data)
        
        # Display fitting status
        st.subheader("Model Fitting Status")
        for model, (success, message) in fitting_status.items():
            if success:
                st.success(f"{model.upper()}: {message}")
            else:
                st.warning(f"{model.upper()}: {message}")
        
        # Continue with analysis only if at least one model converged
        if any(status[0] for status in fitting_status.values()):
            # Calculate CLV
            lf_data, model_metrics = calculator.calculate_clv(
                lf_data,
                time_horizon=params['prediction_period'],
                discount_rate=params['discount_rate']
            )
            
            # Display model diagnostics
            st.subheader("Model Diagnostics")
            for model, metrics in model_metrics.items():
                st.write(f"{model.upper()} Metrics:")
                for metric, value in metrics.items():
                    st.write(f"- {metric}: {value:.4f}")
            
            # Continue with rest of the analysis...
            
        else:
            st.error("Unable to fit any models. Please try adjusting the data preprocessing steps or contact support.")
            
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
