import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
import traceback
import sys

class DataPreprocessor:
    def __init__(self):
        self.data_path = None

    def set_data_path(self, data_path):
        """Set the data directory path"""
        if not os.path.isabs(data_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(os.path.dirname(current_dir), data_path)
        else:
            self.data_path = data_path

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        print(f"✅ Data path set to: {self.data_path}")

    def clean_amount_column(self, df, amount_col='amount'):
        df = df.copy()
        if amount_col in df.columns:
            df[amount_col] = df[amount_col].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        return df

    def clean_date_columns(self, df, date_cols):
        df = df.copy()
        for date_col in date_cols:
            if date_col in df.columns:
                if date_col in ['expires', 'acct_open_date']:
                    df[date_col] = df[date_col].astype(str) + '/01'
                    df[date_col] = pd.to_datetime(df[date_col], format='%m/%Y/%d', errors='coerce')
                elif date_col == 'date':
                    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df

    def get_user_list(self):
        users_file = os.path.join(self.data_path, "users_cards.csv")
        if not os.path.exists(users_file):
            raise ValueError(f"Users file not found: {users_file}")

        users_df = pd.read_csv(users_file, usecols=['id', 'address'], nrows=1000)
        users_df['id'] = pd.to_numeric(users_df['id'], errors='coerce').astype('Int64')
        users_df = users_df.dropna(subset=['id'])

        users_list = [(int(row['id']), f"User {int(row['id'])} - {row['address']}") 
                      for _, row in users_df.iterrows()]
        return users_list

    def load_user_specific_data(self, user_id):
        if not self.data_path:
            raise ValueError("Data path not set. Call set_data_path() first.")

        try:
            print(f"\n📦 Loading user data for user ID: {user_id}")
            users_file = os.path.join(self.data_path, "users_cards.csv")
            print(f"📄 Reading users file: {users_file}")
            users_df = pd.read_csv(users_file)

            users_df['id'] = pd.to_numeric(users_df['id'], errors='coerce')
            user_mask = users_df['id'] == int(user_id)
            users_df = users_df.loc[user_mask].copy()
            print(f"✅ User rows after filter: {len(users_df)}")
            if users_df.empty:
                raise ValueError(f"User {user_id} not found")

            print("🧹 Cleaning user income and debt columns...")
            users_df = self.clean_amount_column(users_df, 'per_capita_income')
            if 'yearly_income' in users_df.columns:
                users_df['yearly_income'] = pd.to_numeric(users_df['yearly_income'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            if 'total_debt' in users_df.columns:
                users_df['total_debt'] = pd.to_numeric(users_df['total_debt'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

            print("🔢 Converting user numeric fields...")
            numeric_cols = ['id', 'current_age', 'retirement_age', 'birth_year', 'birth_month', 'latitude', 'longitude', 'credit_score', 'num_credit_cards']
            for col in numeric_cols:
                if col in users_df.columns:
                    if col in ['latitude', 'longitude', 'per_capita_income', 'yearly_income', 'total_debt']:
                        users_df[col] = pd.to_numeric(users_df[col], errors='coerce')
                    else:
                        users_df[col] = pd.to_numeric(users_df[col], errors='coerce').astype('Int64')

            print("💳 Loading cards data...")
            cards_file = os.path.join(self.data_path, "cards_data.csv")
            if os.path.exists(cards_file):
                cards_df = pd.read_csv(cards_file)
                cards_df['client_id'] = pd.to_numeric(cards_df['client_id'], errors='coerce')
                cards_df = cards_df.loc[cards_df['client_id'] == int(user_id)].copy()
                cards_df = self.clean_amount_column(cards_df, 'credit_limit')
                for col in ['id', 'client_id', 'cvv', 'num_cards_issued', 'year_pin_last_changed']:
                    if col in cards_df.columns:
                        cards_df[col] = pd.to_numeric(cards_df[col], errors='coerce').astype('Int64')
                cards_df = self.clean_date_columns(cards_df, ['expires', 'acct_open_date'])
            else:
                cards_df = pd.DataFrame()
            print(f"✅ Cards loaded: {len(cards_df)}")

            print("💰 Loading transactions data...")
            transactions_file = os.path.join(self.data_path, "transaction_data.csv")
            if os.path.exists(transactions_file):
                transactions_df = pd.read_csv(transactions_file)
                transactions_df['client_id'] = pd.to_numeric(transactions_df['client_id'], errors='coerce')
                transactions_df = transactions_df.loc[transactions_df['client_id'] == int(user_id)].copy()
                transactions_df = self.clean_amount_column(transactions_df, 'amount')
                for col in ['id', 'client_id', 'card_id', 'merchant_id', 'zip', 'mcc']:
                    if col in transactions_df.columns:
                        if col in ['amount', 'zip']:
                            transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')
                        else:
                            transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce').astype('Int64')
                transactions_df = self.clean_date_columns(transactions_df, ['date'])

                if 'date' in transactions_df.columns and not transactions_df.empty:
                    date_col = transactions_df['date']
                    if date_col.notna().any():
                        transactions_df['year'] = date_col.dt.year
                        transactions_df['month'] = date_col.dt.month
                        transactions_df['day'] = date_col.dt.day
                    else:
                        transactions_df['year'] = pd.NA
                        transactions_df['month'] = pd.NA
                        transactions_df['day'] = pd.NA
                else:
                    transactions_df['year'] = pd.NA
                    transactions_df['month'] = pd.NA
                    transactions_df['day'] = pd.NA
            else:
                transactions_df = pd.DataFrame()
            print(f"✅ Transactions loaded: {len(transactions_df)}")

            print("📋 Loading MCC codes...")
            mcc_file = os.path.join(self.data_path, "mcc_codes.csv")
            if os.path.exists(mcc_file):
                mcc_df = pd.read_csv(mcc_file)
                if 'mcc' in mcc_df.columns:
                    mcc_df['mcc'] = pd.to_numeric(mcc_df['mcc'], errors='coerce').astype('Int64')
            else:
                mcc_df = pd.DataFrame()

            print("🚨 Loading fraud labels...")
            fraud_file = os.path.join(self.data_path, "train_fraud_labels.csv")
            if not transactions_df.empty and 'id' in transactions_df.columns:
                user_transaction_ids = transactions_df['id'].dropna()
                if len(user_transaction_ids) > 0:
                    user_transaction_ids = user_transaction_ids.astype(int).tolist()
                else:
                    user_transaction_ids = []
            else:
                user_transaction_ids = []

            if len(user_transaction_ids) > 0 and os.path.exists(fraud_file):
                fraud_df = pd.read_csv(fraud_file)
                fraud_df['transaction_id'] = pd.to_numeric(fraud_df['transaction_id'], errors='coerce')
                fraud_df = fraud_df.loc[fraud_df['transaction_id'].isin(user_transaction_ids)].copy()
                if 'transaction_id' in fraud_df.columns:
                    fraud_df['transaction_id'] = fraud_df['transaction_id'].astype('Int64')
                if 'is_fraud' in fraud_df.columns:
                    fraud_df['is_fraud'] = fraud_df['is_fraud'].astype(bool)
            else:
                fraud_df = pd.DataFrame({
                    'transaction_id': pd.Series([], dtype='Int64'),
                    'is_fraud': pd.Series([], dtype=bool)
                })

            print("✅ All data loaded and cleaned successfully.\n")
            return {
                'users': users_df,
                'cards': cards_df,
                'transactions': transactions_df,
                'mcc_codes': mcc_df,
                'fraud_labels': fraud_df
            }

        except Exception as e:
            print(f"❌ Error loading data for user {user_id}: {str(e)}")
            traceback.print_exc()
            return {
                'users': pd.DataFrame(),
                'cards': pd.DataFrame(),
                'transactions': pd.DataFrame(),
                'mcc_codes': pd.DataFrame(),
                'fraud_labels': pd.DataFrame()
            }

    def close_spark(self):
        pass

# ---------- MAIN for direct script execution ----------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/data_preprocessing.py <data_path> <user_id>")
        sys.exit(1)

    data_path = sys.argv[1]
    user_id = sys.argv[2]

    processor = DataPreprocessor()
    processor.set_data_path(data_path)
    data = processor.load_user_specific_data(user_id)
