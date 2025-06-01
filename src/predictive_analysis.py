import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PredictiveAnalyzer:
    """Advanced predictive analytics for spending patterns and anomaly detection"""
    
    def __init__(self, dataframes):
        self.cards_df = dataframes.get('cards', pd.DataFrame())
        self.transactions_df = dataframes.get('transactions', pd.DataFrame())
        self.users_df = dataframes.get('users', pd.DataFrame())
        self.mcc_df = dataframes.get('mcc_codes', pd.DataFrame())
        self.fraud_df = dataframes.get('fraud_labels', pd.DataFrame())
    
    def get_comparison_metrics(self, user_id, time_period):
        """Enhanced comparison metrics for spending analysis"""
        if self.transactions_df.empty:
            return {}
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return {}
        
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        # Calculate current and previous periods
        days = {'1_month': 30, '3_months': 90, '6_months': 180, '1_year': 365}.get(time_period, 90)
        current_start = latest_date - pd.Timedelta(days=days)
        previous_start = latest_date - pd.Timedelta(days=days * 2)
        previous_end = latest_date - pd.Timedelta(days=days)
        
        current_period = user_transactions[user_transactions['date'] >= current_start]
        previous_period = user_transactions[
            (user_transactions['date'] >= previous_start) & 
            (user_transactions['date'] < previous_end)
        ]
        
        current_total = current_period['amount'].sum() if not current_period.empty else 0
        previous_total = previous_period['amount'].sum() if not previous_period.empty else 0
        
        current_count = len(current_period)
        previous_count = len(previous_period)
        
        spending_change = ((current_total - previous_total) / previous_total * 100) if previous_total > 0 else 0
        transaction_change = ((current_count - previous_count) / previous_count * 100) if previous_count > 0 else 0
        
        return {
            'current_spending': current_total,
            'previous_spending': previous_total,
            'spending_change_pct': spending_change,
            'current_transactions': current_count,
            'previous_transactions': previous_count,
            'transaction_change_pct': transaction_change,
            'avg_transaction_current': current_total / current_count if current_count > 0 else 0,
            'avg_transaction_previous': previous_total / previous_count if previous_count > 0 else 0
        }
    
    def get_spending_trends(self, user_id, time_period='1_year'):
        """Get spending trends over time for predictive analysis"""
        if self.transactions_df.empty:
            return pd.DataFrame()
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return pd.DataFrame()
        
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        # Filter by time period
        if time_period == '1_year':
            start_date = latest_date - pd.Timedelta(days=365)
        elif time_period == '6_months':
            start_date = latest_date - pd.Timedelta(days=180)
        else:
            start_date = latest_date - pd.Timedelta(days=90)
        
        filtered_transactions = user_transactions[user_transactions['date'] >= start_date]
        
        # Add categories
        if not self.mcc_df.empty and 'mcc' in filtered_transactions.columns:
            filtered_transactions['mcc'] = filtered_transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            filtered_transactions = filtered_transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            filtered_transactions['category'] = filtered_transactions['category'].fillna('Other')
        else:
            filtered_transactions['category'] = 'Other'
        
        # Create month-year grouping
        filtered_transactions['month_year'] = filtered_transactions['date'].dt.to_period('M')
        
        # Group by month and category
        monthly_trends = filtered_transactions.groupby(['month_year', 'category']).agg({
            'amount': 'sum'
        }).reset_index()
        
        monthly_trends['month_year_str'] = monthly_trends['month_year'].astype(str)
        
        return monthly_trends
    
    def detect_anomalies(self, user_id, time_period='3_months'):
        """Detect spending anomalies based on transaction patterns"""
        if self.transactions_df.empty:
            return {}
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return {}
        
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        # Filter by time period
        days = {'1_month': 30, '3_months': 90, '6_months': 180}.get(time_period, 90)
        start_date = latest_date - pd.Timedelta(days=days)
        period_transactions = user_transactions[user_transactions['date'] >= start_date]
        
        if period_transactions.empty:
            return {}
        
        # Calculate anomaly thresholds
        mean_amount = period_transactions['amount'].mean()
        std_amount = period_transactions['amount'].std()
        threshold = mean_amount + (2 * std_amount)  # 2 standard deviations
        
        # Detect amount anomalies
        anomalies = period_transactions[period_transactions['amount'] > threshold]
        
        # Calculate spending velocity anomalies (transactions per day)
        daily_transaction_counts = period_transactions.groupby(period_transactions['date'].dt.date).size()
        mean_daily_transactions = daily_transaction_counts.mean()
        std_daily_transactions = daily_transaction_counts.std()
        
        velocity_threshold = mean_daily_transactions + (2 * std_daily_transactions)
        velocity_anomalies = daily_transaction_counts[daily_transaction_counts > velocity_threshold]
        
        return {
            'amount_anomalies': len(anomalies),
            'anomaly_threshold': threshold,
            'velocity_anomalies': len(velocity_anomalies),
            'velocity_threshold': velocity_threshold,
            'anomaly_transactions': anomalies[['date', 'amount', 'merchant_id']].to_dict('records') if not anomalies.empty else [],
            'total_anomaly_value': anomalies['amount'].sum() if not anomalies.empty else 0
        }
    
    def predict_spending(self, user_id, months_ahead=3):
        """Predict future spending based on historical trends"""
        if self.transactions_df.empty:
            return {}
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return {}
        
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        
        # Group by month for trend analysis
        monthly_spending = user_transactions.groupby(
            user_transactions['date'].dt.to_period('M')
        )['amount'].sum().reset_index()
        
        if len(monthly_spending) < 3:
            return {'error': 'Insufficient data for prediction'}
        
        # Simple linear trend calculation
        monthly_spending['month_num'] = range(len(monthly_spending))
        spending_values = monthly_spending['amount'].values
        month_numbers = monthly_spending['month_num'].values
        
        # Calculate linear regression coefficients
        slope, intercept = np.polyfit(month_numbers, spending_values, 1)
        
        # Generate predictions
        predictions = []
        last_month_num = month_numbers[-1]
        
        for i in range(1, months_ahead + 1):
            predicted_amount = slope * (last_month_num + i) + intercept
            predicted_amount = max(0, predicted_amount)  # Ensure non-negative
            predictions.append({
                'month': f'Month +{i}',
                'predicted_spending': predicted_amount
            })
        
        # Calculate trend metrics
        recent_avg = spending_values[-3:].mean() if len(spending_values) >= 3 else spending_values.mean()
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'predictions': predictions,
            'trend_direction': trend_direction,
            'trend_slope': slope,
            'recent_average': recent_avg,
            'confidence': 'low' if len(monthly_spending) < 6 else 'medium' if len(monthly_spending) < 12 else 'high'
        }
    
    def get_merchant_insights(self, user_id, time_period='3_months'):
        """Enhanced merchant insights for predictive analysis"""
        if self.transactions_df.empty:
            return {}
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return {}
        
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        # Filter by time period
        days = {'1_month': 30, '3_months': 90, '6_months': 180}.get(time_period, 90)
        start_date = latest_date - pd.Timedelta(days=days)
        period_transactions = user_transactions[user_transactions['date'] >= start_date]
        
        if period_transactions.empty:
            return {}
        
        # Basic merchant analysis
        merchant_stats = period_transactions.groupby('merchant_id').agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'date': ['min', 'max']
        }).round(2)
        
        merchant_stats.columns = ['total_spent', 'transaction_count', 'avg_amount', 'std_amount', 
                                'first_transaction', 'last_transaction']
        merchant_stats = merchant_stats.reset_index()
        
        # Calculate consistency scores
        merchant_stats['std_amount'] = merchant_stats['std_amount'].fillna(0)
        cv = np.where(
            merchant_stats['avg_amount'] > 0,
            merchant_stats['std_amount'] / merchant_stats['avg_amount'],
            0
        )
        merchant_stats['consistency_score'] = (1 / (1 + cv)).round(3)
        
        # Calculate frequency scores
        period_days = (latest_date - start_date).days
        merchant_stats['frequency_score'] = (merchant_stats['transaction_count'] / period_days * 30).round(2)
        
        # Add merchant names (simplified)
        merchant_stats['merchant_name'] = merchant_stats['merchant_id'].apply(
            lambda x: f"Merchant {x}"
        )
        
        # Sort by total spending
        merchant_stats = merchant_stats.sort_values('total_spent', ascending=False)
        
        # Identify subscription patterns (for time periods > 1 month)
        if time_period != '1_month':
            subscriptions = merchant_stats[
                (merchant_stats['consistency_score'] > 0.8) &
                (merchant_stats['frequency_score'] >= 0.8) &
                (merchant_stats['transaction_count'] >= 2)
            ].copy()
            
            if not subscriptions.empty:
                subscriptions['estimated_monthly_cost'] = (
                    subscriptions['total_spent'] / subscriptions['transaction_count']
                ).round(2)
            else:
                subscriptions = pd.DataFrame()
        else:
            subscriptions = {
                'insufficient_data': True,
                'message': 'Select other time frame to know about your subscriptions'
            }
        
        return {
            'merchant_stats': merchant_stats,
            'total_merchants': len(merchant_stats),
            'subscription_merchants': subscriptions
        }