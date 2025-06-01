import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.spending_analysis import SpendingAnalyzer
from src.predictive_analysis import PredictiveAnalyzer

def display_overview_dashboard(analyzer, user_id, time_period, start_date=None, end_date=None):
    """Enhanced overview dashboard with key metrics"""
    
    st.subheader("📊 Spending Overview Dashboard")
    
    # Get comparison metrics
    comparison_metrics = analyzer.get_comparison_metrics(user_id, time_period)
    category_data = analyzer.get_category_breakdown(user_id, time_period, start_date, end_date)
    
    if comparison_metrics:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            spending_change = comparison_metrics.get('spending_change_pct', 0)
            st.metric(
                "💰 Total Spending",
                f"${comparison_metrics.get('current_spending', 0):,.2f}",
                delta=f"{spending_change:+.1f}%",
                delta_color="inverse" if spending_change > 0 else "normal"
            )
        
        with col2:
            transaction_change = comparison_metrics.get('transaction_change_pct', 0)
            st.metric(
                "🛒 Transactions",
                f"{comparison_metrics.get('current_transactions', 0):,}",
                delta=f"{transaction_change:+.1f}%"
            )
        
        with col3:
            avg_current = comparison_metrics.get('avg_transaction_current', 0)
            avg_previous = comparison_metrics.get('avg_transaction_previous', 0)
            avg_change = ((avg_current - avg_previous) / avg_previous * 100) if avg_previous > 0 else 0
            st.metric(
                "📊 Avg Transaction",
                f"${avg_current:.2f}",
                delta=f"{avg_change:+.1f}%"
            )
        
        with col4:
            # Calculate spending velocity (transactions per day)
            days_in_period = 30 if time_period == '1_month' else 90
            velocity = comparison_metrics.get('current_transactions', 0) / days_in_period
            st.metric(
                "⚡ Spending Velocity",
                f"{velocity:.1f}/day",
                delta="transactions per day"
            )
        
        with col5:
            # Calculate diversity score (number of categories used)
            diversity_score = len(category_data) if not category_data.empty else 0
            st.metric(
                "🎯 Category Diversity",
                f"{diversity_score}",
                delta="active categories"
            )
    
    st.markdown("---")
    
    # Enhanced visualizations
    if not category_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive spending distribution with drill-down
            fig = px.sunburst(
                category_data.head(10),
                names='category',
                values='total_spent',
                title="Interactive Spending Distribution",
                color='total_spent',
                color_continuous_scale='viridis'
            )
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>" +
                            "Amount: $%{value:,.2f}<br>" +
                            "Percentage: %{percentParent}<br>" +
                            "<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top categories with enhanced metrics
            st.subheader("🏆 Top Categories")
            for i, row in category_data.head(5).iterrows():
                with st.container():
                    st.markdown(f"**{row['category']}**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Amount", f"${row['total_spent']:,.2f}")
                    with col_b:
                        st.metric("Share", f"{row['percentage']:.1f}%")
                    st.progress(row['percentage'] / 100)
                    st.markdown("---")

def display_category_deep_dive(analyzer, user_id, time_period, start_date=None, end_date=None):
    """Deep dive into category analysis"""
    
    st.subheader("🛍️ Category Deep Dive Analysis")
    
    category_data = analyzer.get_category_breakdown(user_id, time_period, start_date, end_date)
    
    if category_data.empty:
        st.warning("No category data available for the selected period.")
        return
    
    # Category selection for detailed analysis
    selected_category = st.selectbox(
        "Select Category for Detailed Analysis",
        options=category_data['category'].tolist(),
        index=0
    )
    
    # Category-specific insights
    category_info = category_data[category_data['category'] == selected_category].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Spent", f"${category_info['total_spent']:,.2f}")
    with col2:
        st.metric("Transactions", f"{category_info['transaction_count']:,}")
    with col3:
        st.metric("Avg Transaction", f"${category_info['avg_transaction']:,.2f}")
    with col4:
        st.metric("% of Total", f"{category_info['percentage']:.1f}%")
    
    # Enhanced category visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category comparison chart
        fig_comparison = px.bar(
            category_data.head(10),
            x='total_spent',
            y='category',
            orientation='h',
            title="Category Spending Comparison",
            color='total_spent',
            color_continuous_scale='plasma',
            text='total_spent'
        )
        fig_comparison.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='inside'
        )
        fig_comparison.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Transaction frequency vs amount scatter
        fig_scatter = px.scatter(
            category_data,
            x='transaction_count',
            y='avg_transaction',
            size='total_spent',
            hover_name='category',
            title="Transaction Patterns by Category",
            labels={
                'transaction_count': 'Number of Transactions',
                'avg_transaction': 'Average Transaction ($)',
                'total_spent': 'Total Spent'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def display_predictive_insights(analyzer, user_id, time_period, show_predictions, show_anomalies):
    """Display predictive insights and anomaly detection"""
    
    st.subheader("🔮 Predictive Insights & Anomaly Detection")
    
    # Initialize predictive analyzer
    predictive_analyzer = PredictiveAnalyzer(analyzer.__dict__)
    
    if show_predictions:
        st.subheader("📈 Spending Predictions")
        
        # Get predictions
        predictions = predictive_analyzer.predict_spending(user_id, months_ahead=3)
        
        if 'error' not in predictions:
            # Display predictions
            pred_data = predictions.get('predictions', [])
            if pred_data:
                pred_df = pd.DataFrame(pred_data)
                
                # Visualization
                fig = go.Figure()
                
                # Add predictions
                fig.add_trace(go.Scatter(
                    x=pred_df['month'],
                    y=pred_df['predicted_spending'],
                    mode='lines+markers',
                    name='Predicted Spending',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="3-Month Spending Predictions",
                    xaxis_title="Time Period",
                    yaxis_title="Spending ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                trend_direction = predictions.get('trend_direction', 'stable')
                recent_avg = predictions.get('recent_average', 0)
                confidence = predictions.get('confidence', 'low')
                
                st.info(f"📊 **Prediction Summary**: Your spending trend is {trend_direction}. "
                       f"Recent average: ${recent_avg:.2f}. Confidence: {confidence}")
        else:
            st.warning(predictions['error'])
    
    if show_anomalies:
        st.subheader("🚨 Anomaly Detection")
        
        anomalies = predictive_analyzer.detect_anomalies(user_id, time_period)
        
        if anomalies:
            amount_anomalies = anomalies.get('amount_anomalies', 0)
            
            if amount_anomalies > 0:
                st.warning(f"🚨 Detected {amount_anomalies} unusual transactions (significantly above average)")
                
                # Show anomaly details
                anomaly_transactions = anomalies.get('anomaly_transactions', [])
                if anomaly_transactions:
                    anomaly_df = pd.DataFrame(anomaly_transactions)
                    st.dataframe(
                        anomaly_df,
                        column_config={
                            "date": "Date",
                            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                            "merchant_id": "Merchant ID"
                        },
                        use_container_width=True
                    )
            else:
                st.success("✅ No unusual spending patterns detected in your recent transactions.")

def display_advanced_spending_analysis(user_dataframes, selected_user_id):
    """Display the advanced spending analysis page with enhanced features"""
    
    st.title("🔍 Advanced Spending Intelligence")
    st.markdown("*Comprehensive analysis with AI-powered insights and predictive analytics*")
    st.markdown("---")
    
    # Initialize analyzers
    spending_analyzer = SpendingAnalyzer(user_dataframes)
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Time period selection with more options
    time_period_options = {
        "Last 30 Days": "1_month",
        "Last 3 Months": "3_months", 
        "Last 6 Months": "6_months",
        "Last Year": "1_year",
        "Custom Range": "custom"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=1
    )
    
    selected_period = time_period_options[selected_period_display]
    
    # Custom date range
    start_date, end_date = None, None
    if selected_period == "custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    
    # Analysis depth selection
    analysis_depth = st.sidebar.selectbox(
        "🔬 Analysis Depth",
        ["Quick Overview", "Standard Analysis", "Deep Dive"],
        index=1
    )
    
    # Feature toggles
    st.sidebar.subheader("📊 Features")
    show_spending_predictions = st.sidebar.checkbox("📈 Spending Predictions", value=False)
    show_anomaly_detection = st.sidebar.checkbox("🚨 Anomaly Detection", value=False)
    
    # Updated main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview Dashboard", 
        "🛍️ Category Deep Dive", 
        "🔮 Predictive Insights"
    ])
    
    with tab1:
        display_overview_dashboard(
            spending_analyzer, selected_user_id, selected_period, start_date, end_date
        )
    
    with tab2:
        display_category_deep_dive(
            spending_analyzer, selected_user_id, selected_period, start_date, end_date
        )
    
    with tab3:
        if show_spending_predictions or show_anomaly_detection:
            display_predictive_insights(
                spending_analyzer, selected_user_id, selected_period,
                show_spending_predictions, show_anomaly_detection
            )
        else:
            st.info("Enable Predictive features in the sidebar to view this analysis.")
    
    # Add helpful note about Merchant Intelligence
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Merchant Intelligence** is available as a dedicated page in the main navigation for comprehensive merchant analysis.")