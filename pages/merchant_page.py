import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from merchant_analysis import MerchantAnalyzer

def display_merchant_analysis_page(user_dataframes, selected_user_id):
    """Display merchant analysis page - FIXED boolean condition"""
    st.title("🏪 Merchant Intelligence Dashboard")
    st.markdown("*Advanced merchant analysis with pattern recognition and loyalty insights*")
    st.markdown("---")
    
    # Initialize merchant analyzer
    merchant_analyzer = MerchantAnalyzer(
        user_dataframes.get('transactions', pd.DataFrame()),
        user_dataframes.get('mcc_codes', pd.DataFrame())
    )
    
    # Time period selection
    time_period_options = {
        "Last Month": "1_month",
        "Last 3 Months": "3_months", 
        "Last 6 Months": "6_months",
        "Last Year": "Last Year"
    }
    
    selected_period_display = st.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=1
    )
    
    selected_period = time_period_options[selected_period_display]
    
    # Get merchant insights
    merchant_insights = merchant_analyzer.get_merchant_insights(selected_user_id, selected_period)
    
    # FIXED: Proper boolean check for merchant data
    if not merchant_insights or merchant_insights.get('merchant_stats') is None or len(merchant_insights.get('merchant_stats', pd.DataFrame())) == 0:
        st.warning("No merchant data available for analysis.")
        return
    
    merchant_stats = merchant_insights['merchant_stats']
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Merchants", merchant_insights['total_merchants'])
    
    with col2:
        concentration = merchant_insights.get('top_merchant_concentration', {})
        st.metric("Top 3 Concentration", f"{concentration.get('top_3_percentage', 0):.1f}%")
    
    with col3:
        volatility = merchant_insights.get('spending_volatility', {})
        st.metric("Spending Volatility", f"{volatility.get('volatility_coefficient', 0):.2f}")
    
    with col4:
        loyalty_ops = merchant_insights.get('loyalty_opportunities', pd.DataFrame())
        st.metric("Loyalty Opportunities", len(loyalty_ops))
    
    # Merchant tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top Merchants", "🔄 Subscriptions", "🎯 Loyalty Opportunities"])
    
    with tab1:
        if len(merchant_stats) > 0:  # FIXED: Use len() check
            # Top merchants visualization
            top_merchants = merchant_stats.head(10)
            
            fig = px.bar(
                top_merchants,
                x='total_spent',
                y='merchant_name',
                orientation='h',
                title="Top Merchants by Spending",
                color='total_spent',
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Merchant details
            st.dataframe(
                top_merchants[['merchant_name', 'total_spent', 'transaction_count', 'consistency_score']],
                use_container_width=True
            )
    
    with tab2:
        subscriptions = merchant_insights.get('subscription_merchants', pd.DataFrame())
        if isinstance(subscriptions, dict) and subscriptions.get('insufficient_data', False):
            st.subheader("🔄 Subscription Analysis")
            
            # Display the warning message
            st.warning(f"⚠️ {subscriptions.get('message', 'Insufficient data for subscription detection')}")
            
            # Show recommendation
            if subscriptions.get('recommendation'):
                st.info(f"💡 {subscriptions['recommendation']}")
            
            # Show helpful tip
            st.markdown("""
            **Why 1 month isn't enough?**
            - Most subscriptions bill monthly (only 1 transaction)
            - Need multiple transactions to detect patterns
            - Recommended: Use 3+ months for accurate detection
            """)
        else:
            if len(subscriptions) > 0:  # FIXED: Use len() check
                st.subheader("🔄 Detected Subscription Services")
                
                total_monthly = subscriptions['estimated_monthly_cost'].sum()
                st.metric("Total Monthly Subscriptions", f"${total_monthly:.2f}")
                
                for _, sub in subscriptions.iterrows():
                    st.info(f"📍 {sub['merchant_name']}: ~${sub['estimated_monthly_cost']:.2f}/month")
            else:
                st.info("No subscription patterns detected.")
    
    with tab3:
        loyalty_ops = merchant_insights.get('loyalty_opportunities', pd.DataFrame())
        
        if len(loyalty_ops) > 0:  # FIXED: Use len() check
            st.subheader("🎯 Top Loyalty Opportunities")
            
            for _, merchant in loyalty_ops.head(5).iterrows():
                with st.expander(f"💳 {merchant['merchant_name']} - Score: {merchant['loyalty_score']:.1f}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Spent", f"${merchant['total_spent']:,.2f}")
                    with col2:
                        st.metric("Visits", f"{merchant['transaction_count']}")
                    with col3:
                        potential_rewards = merchant['total_spent'] * 0.02
                        st.metric("Potential Rewards", f"${potential_rewards:.2f}")
        else:
            st.info("No specific loyalty opportunities identified.")