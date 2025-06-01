import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class MLPredictiveAnalyzer:
    """Machine Learning-based predictive analytics for expense forecasting"""
    
    def __init__(self, dataframes):
        self.cards_df = dataframes.get('cards', pd.DataFrame())
        self.transactions_df = dataframes.get('transactions', pd.DataFrame())
        self.users_df = dataframes.get('users', pd.DataFrame())
        self.mcc_df = dataframes.get('mcc_codes', pd.DataFrame())
        
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        self.trained_models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.scalers = {}
        
    def prepare_features(self, user_id, min_months=6):
        """
        Prepare comprehensive feature set for ML models
        
        Args:
            user_id: User ID for analysis
            min_months: Minimum months of data required
            
        Returns:
            DataFrame with features and targets for ML training
        """
        if self.transactions_df.empty:
            return pd.DataFrame()
        
        # Get user transactions
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return pd.DataFrame()
        
        # Convert date and sort
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        user_transactions = user_transactions.sort_values('date')
        
        # Check if we have enough historical data
        date_range = (user_transactions['date'].max() - user_transactions['date'].min()).days
        if date_range < min_months * 30:
            return pd.DataFrame()
        
        # Add MCC categories
        if not self.mcc_df.empty and 'mcc' in user_transactions.columns:
            user_transactions['mcc'] = user_transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            user_transactions = user_transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            user_transactions['category'] = user_transactions['category'].fillna('Other')
        else:
            user_transactions['category'] = 'Other'
        
        # Create comprehensive time-based features
        user_transactions['year'] = user_transactions['date'].dt.year
        user_transactions['month'] = user_transactions['date'].dt.month
        user_transactions['quarter'] = user_transactions['date'].dt.quarter
        user_transactions['day_of_week'] = user_transactions['date'].dt.dayofweek
        user_transactions['day_of_month'] = user_transactions['date'].dt.day
        user_transactions['week_of_year'] = user_transactions['date'].dt.isocalendar().week
        user_transactions['is_weekend'] = user_transactions['day_of_week'].isin([5, 6]).astype(int)
        user_transactions['is_month_start'] = (user_transactions['day_of_month'] <= 5).astype(int)
        user_transactions['is_month_end'] = (user_transactions['day_of_month'] >= 25).astype(int)
        
        # Add seasonal indicators
        user_transactions['season'] = user_transactions['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Group by month-year for aggregation
        user_transactions['month_year'] = user_transactions['date'].dt.to_period('M')
        
        # Create comprehensive monthly aggregations
        monthly_features = []
        
        for period, group in user_transactions.groupby('month_year'):
            month_data = {
                'period': period,
                'year': group['year'].iloc[0],
                'month': group['month'].iloc[0],
                'quarter': group['quarter'].iloc[0],
                
                # Basic spending metrics
                'total_spending': group['amount'].sum(),
                'transaction_count': len(group),
                'avg_transaction_amount': group['amount'].mean(),
                'median_transaction_amount': group['amount'].median(),
                'std_transaction_amount': group['amount'].std(),
                'max_transaction_amount': group['amount'].max(),
                'min_transaction_amount': group['amount'].min(),
                
                # Category diversity
                'unique_categories': group['category'].nunique(),
                'unique_merchants': group['merchant_id'].nunique(),
                
                # Temporal patterns
                'weekend_spending': group[group['is_weekend'] == 1]['amount'].sum(),
                'weekday_spending': group[group['is_weekend'] == 0]['amount'].sum(),
                'weekend_transaction_count': (group['is_weekend'] == 1).sum(),
                'weekday_transaction_count': (group['is_weekend'] == 0).sum(),
                
                # Month timing patterns
                'month_start_spending': group[group['is_month_start'] == 1]['amount'].sum(),
                'month_end_spending': group[group['is_month_end'] == 1]['amount'].sum(),
                
                # Season
                'season': group['season'].iloc[0]
            }
            
            # Category-specific spending (top 10 categories)
            category_spending = group.groupby('category')['amount'].sum()
            top_categories = ['Grocery Stores, Supermarkets', 'Eating Places and Restaurants', 
                            'Gas Stations', 'Online Retail', 'Department Stores']
            
            for cat in top_categories:
                month_data[f'spending_{cat.lower().replace(" ", "_").replace(",", "")}'] = category_spending.get(cat, 0)
            
            monthly_features.append(month_data)
        
        if not monthly_features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(monthly_features)
        
        # Handle missing values
        feature_df['std_transaction_amount'] = feature_df['std_transaction_amount'].fillna(0)
        
        # Create lagged features (previous months' patterns)
        feature_df = feature_df.sort_values('period')
        
        # Add lagged features for better prediction
        lag_columns = ['total_spending', 'transaction_count', 'avg_transaction_amount']
        for col in lag_columns:
            for lag in [1, 2, 3]:
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)
        
        # Rolling averages
        for col in lag_columns:
            feature_df[f'{col}_rolling_3'] = feature_df[col].rolling(window=3, min_periods=1).mean()
            feature_df[f'{col}_rolling_6'] = feature_df[col].rolling(window=6, min_periods=1).mean()
        
        # Growth rates
        for col in lag_columns:
            feature_df[f'{col}_growth_rate'] = feature_df[col].pct_change()
        
        # Handle categorical variables
        le_season = LabelEncoder()
        feature_df['season_encoded'] = le_season.fit_transform(feature_df['season'])
        
        # Drop rows with too many NaN values (first few months due to lagging)
        feature_df = feature_df.dropna(subset=['total_spending_lag_1', 'total_spending_lag_2'])
        
        return feature_df
    
    def train_models(self, user_id, target_column='total_spending'):
        """
        Train multiple ML models on historical data
        
        Args:
            user_id: User ID for training
            target_column: Column to predict
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare features
        feature_df = self.prepare_features(user_id)
        
        if feature_df.empty or len(feature_df) < 10:
            return {
                'success': False,
                'error': 'Insufficient historical data for training (need at least 10 months)',
                'data_points': len(feature_df) if not feature_df.empty else 0
            }
        
        # Select features for training
        exclude_columns = ['period', 'season', target_column]
        feature_columns = [col for col in feature_df.columns if col not in exclude_columns]
        
        X = feature_df[feature_columns]
        y = feature_df[target_column]
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Store scaler for later use
        self.scalers[user_id] = {
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train and evaluate each model
        results = {}
        self.trained_models[user_id] = {}
        self.model_metrics[user_id] = {}
        self.feature_importance[user_id] = {}
        
        for model_name, model in self.models.items():
            model_results = []
            
            # Cross-validation
            for train_idx, val_idx in tscv.split(X_scaled_df):
                X_train, X_val = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_train, y_train)
                
                # Predict
                y_pred = model_copy.predict(X_val)
                
                # Calculate metrics
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                model_results.append({
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                })
            
            # Average metrics across folds
            avg_metrics = {
                'mae': np.mean([r['mae'] for r in model_results]),
                'mse': np.mean([r['mse'] for r in model_results]),
                'rmse': np.mean([r['rmse'] for r in model_results]),
                'r2': np.mean([r['r2'] for r in model_results])
            }
            
            # Train final model on all data
            final_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            final_model.fit(X_scaled_df, y)
            
            # Store results
            self.trained_models[user_id][model_name] = final_model
            self.model_metrics[user_id][model_name] = avg_metrics
            
            # Feature importance (for tree-based models)
            if hasattr(final_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': final_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[user_id][model_name] = importance_df
            
            results[model_name] = {
                'metrics': avg_metrics,
                'feature_importance': importance_df if hasattr(final_model, 'feature_importances_') else None
            }
        
        # Find best model
        best_model = min(self.model_metrics[user_id].items(), key=lambda x: x[1]['mae'])
        
        return {
            'success': True,
            'best_model': best_model[0],
            'best_metrics': best_model[1],
            'all_results': results,
            'data_points': len(feature_df),
            'feature_count': len(feature_columns),
            'training_period': f"{feature_df['period'].min()} to {feature_df['period'].max()}"
        }
    
    def predict_future_expenses(self, user_id, months_ahead=6, model_name=None):
        """
        Predict future expenses using trained models
        
        Args:
            user_id: User ID for prediction
            months_ahead: Number of months to predict
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if user_id not in self.trained_models:
            return {
                'success': False,
                'error': 'No trained models found. Please train models first.'
            }
        
        # Get feature data
        feature_df = self.prepare_features(user_id)
        if feature_df.empty:
            return {
                'success': False,
                'error': 'Unable to prepare features for prediction'
            }
        
        # Determine which model to use
        if model_name is None:
            model_name = min(self.model_metrics[user_id].items(), key=lambda x: x[1]['mae'])[0]
        
        if model_name not in self.trained_models[user_id]:
            return {
                'success': False,
                'error': f'Model {model_name} not found'
            }
        
        model = self.trained_models[user_id][model_name]
        scaler_info = self.scalers[user_id]
        
        # Get last known period
        last_period = feature_df['period'].max()
        last_row = feature_df[feature_df['period'] == last_period].iloc[0]
        
        predictions = []
        prediction_data = last_row.copy()
        
        for month_offset in range(1, months_ahead + 1):
            # Create future period
            # Create future period  
            try:
        # Method 1: Using pd.DateOffset
                future_period = last_period + pd.DateOffset(months=month_offset)
            except:
                # Method 2: Alternative approach if DateOffset fails
                last_date = last_period.to_timestamp()
                future_date_alt = last_date + pd.DateOffset(months=month_offset)
                future_period = future_date_alt.to_period('M')
            
            future_date = future_period.to_timestamp()
            
            # Update time-based features
            prediction_data['year'] = future_date.year
            prediction_data['month'] = future_date.month
            prediction_data['quarter'] = future_date.quarter
            
            # Update seasonal encoding
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall'}
            season = season_map[future_date.month]
            
            # Encode season (simplified - using same encoding as training)
            season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
            prediction_data['season_encoded'] = season_encoding.get(season, 0)
            
            # Prepare features for prediction
            feature_columns = scaler_info['feature_columns']
            X_pred = prediction_data[feature_columns].values.reshape(1, -1)
            
            # Handle any NaN values
            X_pred = np.nan_to_num(X_pred, nan=np.nanmean(X_pred))
            
            # Scale features
            X_pred_scaled = scaler_info['scaler'].transform(X_pred)
            
            # Make prediction
            prediction = model.predict(X_pred_scaled)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
            # Update lagged features for next prediction
            if month_offset < months_ahead:
                # Shift lagged features
                prediction_data['total_spending_lag_3'] = prediction_data['total_spending_lag_2']
                prediction_data['total_spending_lag_2'] = prediction_data['total_spending_lag_1']
                prediction_data['total_spending_lag_1'] = prediction
                
                # Update rolling averages (simplified)
                recent_values = [prediction_data['total_spending_lag_1'], 
                               prediction_data['total_spending_lag_2'], 
                               prediction_data['total_spending_lag_3']]
                prediction_data['total_spending_rolling_3'] = np.mean([v for v in recent_values if pd.notna(v)])
            
            predictions.append({
                'period': str(future_period),
                'month_year': future_date.strftime('%Y-%m'),
                'predicted_spending': round(prediction, 2),
                'month_name': future_date.strftime('%B %Y'),
                'confidence': self._calculate_confidence(user_id, model_name)
            })
        
        # Calculate trend and totals
        predicted_values = [p['predicted_spending'] for p in predictions]
        trend = 'increasing' if predicted_values[-1] > predicted_values[0] else 'decreasing'
        
        return {
            'success': True,
            'model_used': model_name,
            'predictions': predictions,
            'total_predicted': sum(predicted_values),
            'average_monthly': np.mean(predicted_values),
            'trend_direction': trend,
            'confidence_level': self._calculate_confidence(user_id, model_name)
        }
    
    def _calculate_confidence(self, user_id, model_name):
        """Calculate confidence level based on model performance"""
        if user_id not in self.model_metrics or model_name not in self.model_metrics[user_id]:
            return 'low'
        
        r2_score = self.model_metrics[user_id][model_name]['r2']
        
        if r2_score >= 0.8:
            return 'high'
        elif r2_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def get_model_performance(self, user_id):
        """Get comprehensive model performance metrics"""
        if user_id not in self.model_metrics:
            return {}
        
        performance_data = []
        for model_name, metrics in self.model_metrics[user_id].items():
            performance_data.append({
                'Model': model_name,
                'R² Score': f"{metrics['r2']:.3f}",
                'MAE': f"${metrics['mae']:.2f}",
                'RMSE': f"${metrics['rmse']:.2f}",
                'Performance': 'Excellent' if metrics['r2'] >= 0.8 else 'Good' if metrics['r2'] >= 0.6 else 'Fair'
            })
        
        return pd.DataFrame(performance_data)
    
    def get_feature_importance_analysis(self, user_id, model_name=None):
        """Get feature importance analysis for model interpretability"""
        if user_id not in self.feature_importance:
            return pd.DataFrame()
        
        if model_name is None:
            # Get best performing model
            model_name = min(self.model_metrics[user_id].items(), key=lambda x: x[1]['mae'])[0]
        
        if model_name not in self.feature_importance[user_id]:
            return pd.DataFrame()
        
        return self.feature_importance[user_id][model_name].head(10)
    
    def detect_spending_anomalies(self, user_id, threshold_std=2.5):
        """Detect anomalies in spending patterns using ML techniques"""
        feature_df = self.prepare_features(user_id)
        
        if feature_df.empty or len(feature_df) < 6:
            return {}
        
        # Calculate anomalies based on total spending
        spending = feature_df['total_spending']
        mean_spending = spending.mean()
        std_spending = spending.std()
        
        # Define anomaly threshold
        upper_threshold = mean_spending + (threshold_std * std_spending)
        lower_threshold = max(0, mean_spending - (threshold_std * std_spending))
        
        # Identify anomalies
        anomalies = feature_df[
            (feature_df['total_spending'] > upper_threshold) | 
            (feature_df['total_spending'] < lower_threshold)
        ]
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_periods': anomalies['period'].astype(str).tolist(),
            'anomaly_amounts': anomalies['total_spending'].tolist(),
            'threshold_upper': upper_threshold,
            'threshold_lower': lower_threshold,
            'mean_spending': mean_spending,
            'anomaly_details': anomalies[['period', 'total_spending', 'transaction_count']].to_dict('records')
        }