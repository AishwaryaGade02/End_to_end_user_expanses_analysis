import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import your ML predictive analyzer
from predictive_analysis import MLPredictiveAnalyzer

def display_future_expense_predictions(analyzer, user_id):
    """Display future expense predictions with automatic model training"""
    st.subheader("🔮 Future Expense Predictions")
    
    # Auto-train models in background if not already trained
    if user_id not in analyzer.trained_models:
        with st.spinner("🤖 Training ML models on your data..."):
            training_results = analyzer.train_models(user_id)
            if not training_results['success']:
                st.error(f"❌ Unable to create predictions: {training_results['error']}")
                return
            else:
                st.success(f"✅ ML models trained successfully using {training_results['data_points']} months of data")
    
    # Prediction controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        months_ahead = st.slider("Months to Predict", 3, 12, 6)
    
    with col2:
        # Auto-generate predictions button
        if st.button("🎯 Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                predictions = analyzer.predict_future_expenses(user_id, months_ahead)
                st.session_state['predictions'] = predictions
    
    with col3:
        # Show confidence level if predictions exist
        if 'predictions' in st.session_state and st.session_state['predictions']['success']:
            confidence = st.session_state['predictions']['confidence_level']
            confidence_color = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
            st.metric(f"{confidence_color} Confidence", confidence.title())
    
    # Display predictions
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        
        if predictions['success']:
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Total Predicted", f"${predictions['total_predicted']:,.2f}")
            with col2:
                st.metric("📊 Monthly Average", f"${predictions['average_monthly']:,.2f}")
            with col3:
                trend_icon = "📈" if predictions['trend_direction'] == 'increasing' else "📉" if predictions['trend_direction'] == 'decreasing' else "➡️"
                st.metric(f"{trend_icon} Trend", predictions['trend_direction'].title())
            
            # Main prediction visualization
            pred_data = predictions['predictions']
            pred_df = pd.DataFrame(pred_data)
            
            # Get historical data for context
            # Replace the problematic section in your display_future_expense_predictions function:

# Get historical data for context
        feature_df = analyzer.prepare_features(user_id)
        if not feature_df.empty:
            
            feature_df['datetime'] = pd.to_datetime(feature_df['period'].astype(str))

            # Filter data from 2012 onwards (or any specific start year you want)
            start_year = 2012
            feature_df = feature_df[feature_df['datetime'].dt.year >= start_year]
            
            if feature_df.empty:
                st.warning(f"No data available from {start_year} onwards")
                return
            
            # Prepare historical data with CORRECT datetime formatting
            historical_df = feature_df[['period', 'total_spending', 'datetime']].copy()
            historical_df['month_year'] = historical_df['datetime'].dt.strftime('%Y-%m')  # FIXED: Use 'datetime' instead of 'total_spending'
            historical_df['type'] = 'Historical'
            historical_df = historical_df.rename(columns={'total_spending': 'amount'})
            
            # Prepare predicted data
            pred_viz_df = pred_df[['month_year', 'predicted_spending']].copy()
            pred_viz_df['type'] = 'Predicted'
            pred_viz_df = pred_viz_df.rename(columns={'predicted_spending': 'amount'})
            
            # Convert prediction dates to datetime for proper ordering
            pred_viz_df['datetime'] = pd.to_datetime(pred_viz_df['month_year'])
            
            # Combine for visualization
            historical_for_viz = historical_df[['month_year', 'amount', 'type', 'datetime']]
            predicted_for_viz = pred_viz_df[['month_year', 'amount', 'type', 'datetime']]
                
            viz_df = pd.concat([historical_for_viz, predicted_for_viz])
            viz_df = viz_df.sort_values('datetime')
            
            # Create main prediction chart
            fig = go.Figure()
            
            # Historical spending line
            historical_data = viz_df[viz_df['type'] == 'Historical']
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['datetime'],  # Use datetime for x-axis
                    y=historical_data['amount'],
                    mode='lines+markers',
                    name='Historical Spending',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=6, color='#2E86AB'),
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Amount: $%{y:,.2f}<extra></extra>"
                ))
            
            # Predicted spending line
            predicted_data = viz_df[viz_df['type'] == 'Predicted']
            if not predicted_data.empty:
                fig.add_trace(go.Scatter(
                    x=predicted_data['datetime'],  # Use datetime for x-axis
                    y=predicted_data['amount'],
                    mode='lines+markers',
                    name='Predicted Spending',
                    line=dict(color='#F18F01', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond', color='#F18F01'),
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Predicted: $%{y:,.2f}<extra></extra>"
                ))
                
                # Add confidence bands for predictions only
                confidence_multiplier = {'high': 0.05, 'medium': 0.10, 'low': 0.15}[predictions['confidence_level']]
                upper_bound = predicted_data['amount'] * (1 + confidence_multiplier)
                lower_bound = predicted_data['amount'] * (1 - confidence_multiplier)
                
                # Upper confidence bound
                fig.add_trace(go.Scatter(
                    x=predicted_data['datetime'],  # FIXED: Use 'datetime' instead of 'month_year'
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower confidence bound with fill
                fig.add_trace(go.Scatter(
                    x=predicted_data['datetime'],  # FIXED: Use 'datetime' instead of 'month_year'
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(241, 143, 1, 0.2)',
                    fill='tonexty',
                    name='Confidence Range',
                    hoverinfo='skip'
                ))
            
            # Add vertical line to separate historical from predicted
            if not historical_data.empty and not predicted_data.empty:
                # Use the last historical date for separation
                last_historical_date = historical_data['datetime'].max()
                fig.add_vline(
                x=last_historical_date.timestamp() * 1000,  # Convert to milliseconds for plotly
                line_dash="dot",
                line_color="gray",
                annotation_text="Historical | Predicted",
                annotation_position="top"
            )
            
            # Set proper date range for x-axis
            if not viz_df.empty:
                min_date = viz_df['datetime'].min()
                max_date = viz_df['datetime'].max()
                
                fig.update_layout(
                    title="Future Expense Predictions Based on Historical Patterns",
                    xaxis_title="Time Period",
                    yaxis_title="Monthly Spending ($)",
                    height=600,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        type='date',
                        range=[min_date, max_date],
                        tickformat='%Y-%m',
                        tickangle=45,
                        dtick="M6"  # Show ticks every 6 months
                    )
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions summary table
            st.subheader("📋 Monthly Breakdown")
            
            # Add trend indicators to table
            display_df = pred_df.copy()
            display_df['trend'] = ''
            for i in range(1, len(display_df)):
                current = display_df.iloc[i]['predicted_spending']
                previous = display_df.iloc[i-1]['predicted_spending']
                if current > previous * 1.02:  # 2% threshold
                    display_df.iloc[i, display_df.columns.get_loc('trend')] = '📈'
                elif current < previous * 0.98:
                    display_df.iloc[i, display_df.columns.get_loc('trend')] = '📉'
                else:
                    display_df.iloc[i, display_df.columns.get_loc('trend')] = '➡️'
            
            st.dataframe(
                display_df[['month_name', 'predicted_spending', 'trend']],
                column_config={
                    "month_name": st.column_config.TextColumn("Month"),
                    "predicted_spending": st.column_config.NumberColumn("Predicted Spending", format="$%,.2f"),
                    "trend": st.column_config.TextColumn("Trend", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # AI Insights
            st.subheader("🧠 AI Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if predictions['trend_direction'] == 'increasing':
                    st.warning("📈 **Spending Trend: Increasing**")
                    st.markdown("""
                    **Recommendations:**
                    - Monitor discretionary categories
                    - Set up spending alerts
                    - Review recurring subscriptions
                    """)
                elif predictions['trend_direction'] == 'decreasing':
                    st.success("📉 **Spending Trend: Decreasing**")
                    st.markdown("""
                    **Opportunities:**
                    - Increase savings rate
                    - Consider investments
                    - Build emergency fund
                    """)
                else:
                    st.info("➡️ **Spending Trend: Stable**")
                    st.markdown("""
                    **Maintain:**
                    - Current spending habits
                    - Regular budget reviews
                    - Consistent savings
                    """)
            
            with col2:
                # Seasonal insights
                seasonal_months = ['December', 'November', 'January']  # Holiday seasons
                high_months = [p for p in pred_data if any(month in p['month_name'] for month in seasonal_months)]
                
                if high_months:
                    st.info("🎄 **Seasonal Pattern Detected**")
                    st.markdown("Higher spending predicted during holiday months")
                
                # Budget suggestions
                avg_spending = predictions['average_monthly']
                st.metric("💡 Suggested Monthly Budget", f"${avg_spending * 1.1:,.2f}")
                st.caption("10% buffer added for unexpected expenses")
        
        else:
            st.error(f"❌ Prediction failed: {predictions['error']}")

def display_anomaly_detection_section(analyzer, user_id):
    """Display anomaly detection with simplified interface"""
    st.subheader("🚨 Spending Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Automatically detect unusual spending patterns that significantly deviate 
        from your normal behavior using advanced statistical analysis.
        """)
    
    with col2:
        sensitivity = st.selectbox(
            "Detection Sensitivity",
            options=[("High Sensitivity", 1.5), ("Medium Sensitivity", 2.0), ("Low Sensitivity", 2.5)],
            index=1,
            format_func=lambda x: x[0]
        )
        
        if st.button("🔍 Analyze Patterns"):
            with st.spinner("Analyzing spending patterns..."):
                anomalies = analyzer.detect_spending_anomalies(user_id, sensitivity[1])
                st.session_state['anomalies'] = anomalies
    
    # Display results
    if 'anomalies' in st.session_state:
        anomalies = st.session_state['anomalies']
        
        if anomalies:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🚨 Anomalies Found", anomalies['anomaly_count'])
            with col2:
                st.metric("📊 Average Spending", f"${anomalies['mean_spending']:,.2f}")
            with col3:
                st.metric("⚠️ Alert Threshold", f"${anomalies['threshold_upper']:,.2f}")
            
            if anomalies['anomaly_count'] > 0:
                st.warning(f"⚠️ Found {anomalies['anomaly_count']} unusual spending periods")
                
                # Visualization
                feature_df = analyzer.prepare_features(user_id)
                if not feature_df.empty:
                    fig = go.Figure()
                    
                    # All spending data
                    fig.add_trace(go.Scatter(
                        x=feature_df['period'].astype(str),
                        y=feature_df['total_spending'],
                        mode='lines+markers',
                        name='Monthly Spending',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=6, color='#2E86AB')
                    ))
                    
                    # Highlight anomalies
                    anomaly_data = feature_df[feature_df['period'].astype(str).isin(anomalies['anomaly_periods'])]
                    if not anomaly_data.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_data['period'].astype(str),
                            y=anomaly_data['total_spending'],
                            mode='markers',
                            name='Anomalous Spending',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='x-thin',
                                line=dict(width=2, color='darkred')
                            )
                        ))
                    
                    # Add threshold lines
                    fig.add_hline(
                        y=anomalies['threshold_upper'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Upper Alert Threshold",
                        annotation_position="bottom right"
                    )
                    
                    fig.add_hline(
                        y=anomalies['mean_spending'],
                        line_dash="dot",
                        line_color="green",
                        annotation_text="Average Spending",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title="Spending Pattern Analysis - Anomaly Detection",
                        xaxis_title="Time Period",
                        yaxis_title="Monthly Spending ($)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details table
                if anomalies['anomaly_details']:
                    st.subheader("🔍 Anomaly Details")
                    anomaly_df = pd.DataFrame(anomalies['anomaly_details'])
                    anomaly_df['period'] = anomaly_df['period'].astype(str)
                    anomaly_df['deviation'] = ((anomaly_df['total_spending'] - anomalies['mean_spending']) / anomalies['mean_spending'] * 100).round(1)
                    
                    st.dataframe(
                        anomaly_df,
                        column_config={
                            "period": "Period",
                            "total_spending": st.column_config.NumberColumn("Spending", format="$%,.2f"),
                            "transaction_count": "Transactions",
                            "deviation": st.column_config.NumberColumn("Deviation %", format="%.1f%%")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.success("✅ No anomalies detected - Your spending patterns are consistent!")
                
                # Still show the spending pattern chart
                feature_df = analyzer.prepare_features(user_id)
                if not feature_df.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=feature_df['period'].astype(str),
                        y=feature_df['total_spending'],
                        mode='lines+markers',
                        name='Monthly Spending',
                        line=dict(color='#28a745', width=3),
                        marker=dict(size=8, color='#28a745'),
                        fill='tonexty'
                    ))
                    
                    fig.add_hline(
                        y=anomalies['mean_spending'],
                        line_dash="dot",
                        line_color="green",
                        annotation_text="Average Spending"
                    )
                    
                    fig.update_layout(
                        title="Your Consistent Spending Pattern",
                        xaxis_title="Time Period",
                        yaxis_title="Monthly Spending ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_ml_predictive_analytics_page(user_dataframes, selected_user_id):
    """Simplified ML-based predictive analytics page focusing on predictions and anomalies"""
    
    st.title("🤖 AI Expense Forecasting")
    st.markdown("*Predict your future spending patterns and detect unusual behavior*")
    st.markdown("---")
    
    # Initialize ML analyzer
    analyzer = MLPredictiveAnalyzer(user_dataframes)
    
    # Quick data validation
    if user_dataframes.get('transactions', pd.DataFrame()).empty:
        st.error("❌ No transaction data available. Please ensure transaction data is loaded.")
        return
    
    user_transactions = user_dataframes['transactions'][
        user_dataframes['transactions']['client_id'] == selected_user_id
    ]
    
    if user_transactions.empty:
        st.warning(f"⚠️ No transactions found for user {selected_user_id}")
        return
    
    # Quick data overview
    transaction_count = len(user_transactions)
    date_range = (pd.to_datetime(user_transactions['date']).max() - 
                 pd.to_datetime(user_transactions['date']).min()).days
    
    if date_range < 180:
        st.warning("⚠️ Limited data may affect prediction accuracy. Best results with 6+ months of data.")
    
    # Main content - just two tabs
    tab1, tab2 = st.tabs([
        "🔮 Future Predictions",
        "🚨 Anomaly Detection"
    ])
    
    with tab1:
        display_future_expense_predictions(analyzer, selected_user_id)
    
    with tab2:
        display_anomaly_detection_section(analyzer, selected_user_id)
    
    # Simple footer
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 AI Predictions:**
        - Analyzes your historical spending patterns
        - Uses machine learning to forecast future expenses
        - Considers seasonality and trends
        - Provides confidence levels for reliability
        """)
    
    with col2:
        st.markdown("""
        **🚨 Anomaly Detection:**
        - Identifies unusual spending periods
        - Compares against your normal patterns
        - Adjustable sensitivity levels
        - Helps spot potential issues early
        """)