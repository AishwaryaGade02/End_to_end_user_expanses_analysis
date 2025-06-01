import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.overview_analysis import OptimizedKPICalculator

def display_user_profile(profile):
    """Display user profile information"""
    if not profile:
        st.error("User profile not found")
        return
    
    st.subheader("👤 User Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{profile['age']} years")
        st.metric("Gender", profile['gender'])
    
    with col2:
        st.metric("Credit Score", profile['credit_score'])
        st.metric("Number of Cards", profile['num_credit_cards'])
    
    with col3:
        st.metric("Yearly Income", f"${profile['yearly_income']:,.0f}")
        st.metric("Total Debt", f"${profile['total_debt']:,.0f}")
    
    with col4:
        st.metric("Address", profile['address'], label_visibility="visible")
    
    # Display cards information
    if profile['cards']:
        st.subheader("💳 User's Cards")
        cards_df = pd.DataFrame(profile['cards'])
        st.dataframe(cards_df, use_container_width=True)

def display_kpi_cards(kpis):
    """Display KPI cards"""
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Monthly Spending",
            value=f"${kpis['total_monthly_spending']:,.2f}",
            delta=None
        )
    
    with col2:
        utilization = kpis['credit_utilization']
        st.metric(
            label="📈 Credit Utilization",
            value=f"{utilization:.1f}%",
            delta=f"{'Good' if utilization < 30 else 'High'}"
        )
    
    with col3:
        st.metric(
            label="💸 Est. Interest Paid",
            value=f"${kpis['interest_paid']:,.2f}",
            delta="Last 12 months"
        )
    
    with col4:
        st.metric(
            label="🎁 Rewards Earned",
            value=f"${kpis['rewards_earned']:,.2f}",
            delta="Last 12 months"
        )
    
    with col5:
        credit_score = kpis['credit_score']
        score_status = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair"
        st.metric(
            label="🏆 Credit Score",
            value=credit_score,
            delta=score_status
        )

def display_spending_breakdown(spending_data):
    """Display spending breakdown by category with bar chart showing top 10 categories"""
    if not spending_data:
        st.info("No spending data available for this user")
        return
    
    st.subheader("🛍️ Spending by Category (Last 3 Months)")
    
    # Create DataFrame and calculate percentages
    df = pd.DataFrame(spending_data)
    
    if df.empty:
        st.info("No spending data to display")
        return
    
    # Calculate total spending for percentage calculation
    total_spending = df['amount'].sum()
    df['percentage'] = (df['amount'] / total_spending * 100).round(1)
    
    # Get top 10 categories
    top_10_df = df.head(10)
    
    # Main chart section (full width)
    # Create horizontal bar chart with better clarity
    fig = px.bar(
        top_10_df, 
        x='percentage',
        y='category',
        orientation='h',
        title="Top 10 Spending Categories",
        labels={'percentage': 'Percentage of Total Spending (%)', 'category': 'Category'},
        color='percentage',
        color_continuous_scale='viridis',
        text='percentage'
    )
    
    # Update layout for better readability
    fig.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>" +
                     "Amount: $%{customdata[0]:,.2f}<br>" +
                     "Percentage: %{x:.1f}%<br>" +
                     "Transactions: %{customdata[1]}<br>" +
                     "<extra></extra>",
        customdata=top_10_df[['amount', 'transactions']].values
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False,
        xaxis_title="Percentage of Total Spending (%)",
        yaxis_title="Category"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category Details section below the chart
    st.subheader("📊 Category Details")
    
    # Create columns for side-by-side category details
    num_categories = min(5, len(top_10_df))
    cols = st.columns([2, 2, 2, 2, 2])
    
    # Display top categories side by side
    for i, (col, (_, row)) in enumerate(zip(cols, top_10_df.head(num_categories).iterrows())):
        with col:
            # Category card
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; color: #1f77b4;">#{i+1} {row['category'][:20]}{'...' if len(row['category']) > 20 else ''}</h4>
                <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: bold;">${row['amount']:,.2f}</p>
                <p style="margin: 0; color: #666;">
                    {row['percentage']:.1f}% • {row['transactions']} transactions
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual representation
            st.progress(row['percentage'] / 100)
    
    # Summary statistics section
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Total Categories", len(df))
    
    with summary_col2:
        st.metric("Total Spending", f"${total_spending:,.2f}")
    
    with summary_col3:
        top_5_percentage = top_10_df.head(5)['percentage'].sum()
        st.metric("Top 5 Share", f"{top_5_percentage:.1f}%")
    
    with summary_col4:
        avg_transaction = df['amount'].sum() / df['transactions'].sum() if df['transactions'].sum() > 0 else 0
        st.metric("Avg Transaction", f"${avg_transaction:.2f}")
    
    

def display_monthly_trend(trend_data):
    """Display monthly spending trend"""
    if not trend_data:
        st.info("No trend data available")
        return
    
    st.subheader("📈 Monthly Spending Trend")
    
    df = pd.DataFrame(trend_data)
    
    if not df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['spending'],
            mode='lines+markers',
            name='Monthly Spending',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Spending Trend Over Time",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data to display")

def display_fraud_risk(fraud_score):
    """Display fraud risk indicator"""
    st.subheader("🔒 Fraud Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Create gauge chart for fraud risk
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fraud_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score (%)"},
            delta = {'reference': 5},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if fraud_score < 5:
            st.success("✅ Low fraud risk - Your account shows normal transaction patterns")
        elif fraud_score < 15:
            st.warning("⚠️ Moderate fraud risk - Monitor your account regularly")
        else:
            st.error("🚨 High fraud risk - Consider reviewing recent transactions")
        
        st.markdown("""
        **Fraud Prevention Tips:**
        - Monitor your accounts regularly
        - Set up transaction alerts
        - Use secure payment methods
        - Report suspicious activity immediately
        """)

def display_overview_page(user_dataframes, selected_user_id):
    """Display the main overview page"""
    
    # Initialize KPI calculator with user-specific data
    kpi_calculator = OptimizedKPICalculator(user_dataframes)
    
    # Get user profile and KPIs
    user_profile = kpi_calculator.get_user_profile(selected_user_id)
    
    if not user_profile:
        st.error(f"No profile found for User {selected_user_id}")
        return
    
    # Display user profile
    display_user_profile(user_profile)
    
    st.markdown("---")
    
    # Get and display KPIs
    kpis = kpi_calculator.get_all_kpis(selected_user_id)
    display_kpi_cards(kpis)
    
    st.markdown("---")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        display_spending_breakdown(kpis['spending_by_category'])
    
    with col2:
        display_fraud_risk(kpis['fraud_risk_score'])
    
    st.markdown("---")
    
    # Monthly trend
    display_monthly_trend(kpis['monthly_trend'])