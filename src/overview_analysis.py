import pandas as pd
from datetime import datetime, timedelta

class OptimizedKPICalculator:
    def __init__(self, dataframes):
        self.cards_df = dataframes.get('cards', pd.DataFrame())
        self.transactions_df = dataframes.get('transactions', pd.DataFrame())
        self.users_df = dataframes.get('users', pd.DataFrame())
        self.mcc_df = dataframes.get('mcc_codes', pd.DataFrame())
        self.fraud_df = dataframes.get('fraud_labels', pd.DataFrame())
    
    def get_user_profile(self, user_id):
        """Get user profile information"""
        if self.users_df.empty:
            return None
            
        user_profile = self.users_df[self.users_df['id'] == user_id]
        
        if user_profile.empty:
            return None
        
        user = user_profile.iloc[0]
        
        # Get user's cards
        if not self.cards_df.empty:
            user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        else:
            user_cards = pd.DataFrame()
        
        profile = {
            'user_id': user['id'],
            'age': user['current_age'],
            'gender': user['gender'],
            'address': user['address'],
            'yearly_income': user['yearly_income'],
            'total_debt': user['total_debt'],
            'credit_score': user['credit_score'],
            'num_credit_cards': user['num_credit_cards'],
            'cards': []
        }
        
        for _, card in user_cards.iterrows():
            profile['cards'].append({
                'card_id': card['id'],
                'card_brand': card['card_brand'],
                'card_type': card['card_type'],
                'credit_limit': card['credit_limit'],
                'expires': card['expires']
            })
        
        return profile
    
    def calculate_monthly_spending(self, user_id, months_back=1):
        """Calculate total monthly spending for the user"""
        if self.transactions_df.empty:
            return 0.0
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return 0.0
        
        # Convert date column to datetime if it's not already
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        
        # Get latest date
        latest_date = user_transactions['date'].max()
        start_date = latest_date - pd.Timedelta(days=30 * months_back)
        
        # Filter for spending in the period
        period_spending = user_transactions[
            (user_transactions['date'] >= start_date) &
            (user_transactions['date'] <= latest_date) &
            (user_transactions['amount'] > 0)
        ]
        
        return period_spending['amount'].sum() if not period_spending.empty else 0.0
    
    def calculate_credit_utilization(self, user_id):
        """Calculate average credit utilization percentage"""
        if self.cards_df.empty:
            return 0.0
            
        user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        
        if user_cards.empty:
            return 0.0
        
        total_credit_limit = user_cards['credit_limit'].sum()
        
        if total_credit_limit == 0:
            return 0.0
        
        current_spending = self.calculate_monthly_spending(user_id, 1)
        utilization = (current_spending / total_credit_limit) * 100
        
        return min(utilization, 100.0)
    
    def calculate_interest_paid(self, user_id, months_back=12):
        """Estimate interest paid based on debt and typical credit card APR"""
        if self.users_df.empty:
            return 0.0
            
        user_data = self.users_df[self.users_df['id'] == user_id]
        
        if user_data.empty:
            return 0.0
        
        total_debt = user_data.iloc[0]['total_debt']
        
        if pd.isna(total_debt) or total_debt == 0:
            return 0.0
        
        estimated_annual_interest = total_debt * 0.18
        monthly_interest = estimated_annual_interest / 12
        
        return monthly_interest * months_back
    
    def calculate_rewards_earned(self, user_id, months_back=12):
        """Calculate estimated rewards earned (assuming 1% cashback)"""
        total_spending = 0
        for month in range(months_back):
            monthly_spend = self.calculate_monthly_spending(user_id, month + 1)
            total_spending += monthly_spend
        
        rewards = total_spending * 0.01
        return rewards
    
    def get_spending_by_category(self, user_id, months_back=3):
        """Get spending breakdown by merchant category"""
        if self.transactions_df.empty:
            return []
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return []
        
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        start_date = latest_date - pd.Timedelta(days=30 * months_back)
        
        # Filter transactions
        period_transactions = user_transactions[
            (user_transactions['date'] >= start_date) &
            (user_transactions['amount'] > 0)
        ]
        
        if period_transactions.empty:
            return []
        
        # Join with MCC codes if available
        if not self.mcc_df.empty:
            spending_with_categories = period_transactions.merge(
                self.mcc_df, on='mcc', how='left'
            )
            spending_with_categories['category'] = spending_with_categories['category'].fillna('Other')
        else:
            spending_with_categories = period_transactions.copy()
            spending_with_categories['category'] = 'Other'
        
        # Group by category
        category_spending = spending_with_categories.groupby('category').agg({
            'amount': 'sum',
            'id': 'count'
        }).reset_index()
        
        category_spending.columns = ['category', 'amount', 'transactions']
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        return category_spending.to_dict('records')
    
    def get_fraud_risk_score(self, user_id):
        """Calculate fraud risk score based on transaction patterns"""
        if self.transactions_df.empty:
            return 0.0
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return 0.0
        
        # Join with fraud labels if available
        if not self.fraud_df.empty:
            fraud_analysis = user_transactions.merge(
                self.fraud_df, left_on='id', right_on='transaction_id', how='left'
            )
            
            total_transactions = len(fraud_analysis)
            fraud_transactions = fraud_analysis['is_fraud'].sum() if not fraud_analysis['is_fraud'].isna().all() else 0
            
            if total_transactions == 0:
                return 0.0
            
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            return fraud_percentage
        else:
            return 0.0
    
    def get_monthly_trend(self, user_id, months=6):
        """Get monthly spending trend"""
        if self.transactions_df.empty:
            return []
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return []
        
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        trends = []
        for i in range(months):
            month_start = latest_date - pd.Timedelta(days=30 * (i + 1))
            month_end = latest_date - pd.Timedelta(days=30 * i)
            
            monthly_transactions = user_transactions[
                (user_transactions['date'] >= month_start) &
                (user_transactions['date'] < month_end) &
                (user_transactions['amount'] > 0)
            ]
            
            monthly_spend = monthly_transactions['amount'].sum() if not monthly_transactions.empty else 0.0
            
            trends.append({
                'month': f"{month_start.strftime('%B')} {month_start.year}",
                'spending': monthly_spend
            })
        
        return list(reversed(trends))
    
    def get_all_kpis(self, user_id):
        """Get all KPIs for a user"""
        profile = self.get_user_profile(user_id)
        credit_score = profile['credit_score'] if profile else 0
        
        return {
            'total_monthly_spending': self.calculate_monthly_spending(user_id),
            'credit_utilization': self.calculate_credit_utilization(user_id),
            'interest_paid': self.calculate_interest_paid(user_id),
            'rewards_earned': self.calculate_rewards_earned(user_id),
            'credit_score': credit_score,
            'spending_by_category': self.get_spending_by_category(user_id),
            'fraud_risk_score': self.get_fraud_risk_score(user_id),
            'monthly_trend': self.get_monthly_trend(user_id)
        }