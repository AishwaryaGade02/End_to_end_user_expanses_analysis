import streamlit as st
import sys
import os
import pandas as pd

# Add src and pages directories to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pages'))

from src.data_preprocessing import DataPreprocessor

# Import all page modules
from pages.overview_page import display_overview_page
from pages.spending_page import display_spending_analysis_page
from pages.merchant_page import display_merchant_analysis_page
from pages.rewards_page import display_rewards_optimization_page
from pages.predictive_page import display_advanced_spending_analysis

# Page configuration
st.set_page_config(
    page_title="Credit Card Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stMetric > div {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    .nav-tabs {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .feature-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_preprocessor():
    """Get or create preprocessor instance"""
    preprocessor = DataPreprocessor()
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    preprocessor.set_data_path(data_path)
    return preprocessor

@st.cache_data
def get_user_list():
    """Get list of users for dropdown"""
    try:
        preprocessor = get_preprocessor()
        return preprocessor.get_user_list()
    except Exception as e:
        st.error(f"Error loading user list: {str(e)}")
        return []

@st.cache_data
def load_user_data(user_id):
    """Load data for specific user only"""
    try:
        preprocessor = get_preprocessor()
        spark_dataframes = preprocessor.load_user_specific_data(user_id)
        
        # Convert to pandas for caching and faster operations
        pandas_dataframes = {}
        for key, spark_df in spark_dataframes.items():
            if spark_df.count() > 0:  # Only convert if DataFrame has data
                pandas_dataframes[key] = spark_df.toPandas()
            else:
                # Create empty pandas DataFrame with proper structure
                pandas_dataframes[key] = pd.DataFrame()
        
        return pandas_dataframes
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return None

def main():
    """Main dashboard function with complete navigation"""
    
    # Enhanced Navigation with feature badges
    st.sidebar.title("🚀 Navigation")
    
    # Navigation options with feature descriptions
    page_options = {
        "📊 Overview": {
            "description": "Basic KPIs and user profile",
            "features": ["User Profile", "KPI Cards", "Basic Charts"]
        },
        "🛍️ Spending Analysis": {
            "description": "Detailed spending breakdown",  
            "features": ["Category Analysis", "Time Filtering", "Export Data"]
        },
        "🏪 Merchant Intelligence": {
            "description": "Advanced merchant insights",
            "features": ["Subscription Detection", "Loyalty Analysis", "Merchant Patterns"]
        },
        "💳 Rewards Optimization": {
            "description": "Credit card portfolio optimization",
            "features": ["Portfolio Analysis", "Signup Bonuses", "Break-even Calculations"]
        },
        "🔍 Advanced Intelligence": {
            "description": "AI-powered comprehensive analysis",
            "features": ["Predictive Analytics", "Anomaly Detection", "Deep Insights"]
        }
    }
    
    # Display navigation with descriptions
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        options=list(page_options.keys()),
        index=0,
        format_func=lambda x: f"{x}"
    )
    
    # Show page description and features
    page_info = page_options[page]
    st.sidebar.markdown(f"**{page_info['description']}**")
    
    for feature in page_info['features']:
        st.sidebar.markdown(f"• {feature}")
    
    st.sidebar.markdown("---")
    
    # Load user list
    user_list = get_user_list()
    
    if not user_list:
        st.error("Failed to load user list. Please check your data files.")
        return
    
    # User selection
    st.sidebar.header("👤 User Selection")
    
    user_options = {user[1]: user[0] for user in user_list}
    selected_user_display = st.sidebar.selectbox(
        "Select User for Analysis",
        options=list(user_options.keys()),
        index=0
    )
    selected_user_id = user_options[selected_user_display]
    
    # Load user-specific data with progress indicator
    with st.spinner(f"🔄 Loading data for {selected_user_display.split(' - ')[0]}..."):
        user_dataframes = load_user_data(selected_user_id)
    
    if not user_dataframes:
        st.error(f"❌ Failed to load data for {selected_user_display}")
        return
    
    # Success indicator
    st.sidebar.success(f"✅ Data loaded for User {selected_user_id}")
    
    # Display selected page
    if page == "📊 Overview":
        st.title("💳 Credit Card Optimization Dashboard")
        st.markdown("*Comprehensive financial overview and key performance indicators*")
        st.markdown("---")
        display_overview_page(user_dataframes, selected_user_id)
        
        # Sidebar info for overview
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Overview Features")
        st.sidebar.info("""
        **Current Page Includes:**
        • User profile and demographics
        • Key financial metrics (KPIs)
        • Spending category breakdown
        • Fraud risk assessment
        • Monthly spending trends
        • Credit utilization tracking
        """)
        
    elif page == "🛍️ Spending Analysis":
        display_spending_analysis_page(user_dataframes, selected_user_id)
        
    elif page == "🏪 Merchant Intelligence":
        display_merchant_analysis_page(user_dataframes, selected_user_id)
        
    elif page == "💳 Rewards Optimization":
        display_rewards_optimization_page(user_dataframes, selected_user_id)
        
    elif page == "🔍 Advanced Intelligence":
        display_advanced_spending_analysis(user_dataframes, selected_user_id)
    
    # Performance metrics in sidebar
    if st.sidebar.checkbox("📊 Show Performance Metrics"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Performance Info")
        
        # Data size metrics
        transaction_count = len(user_dataframes.get('transactions', pd.DataFrame()))
        card_count = len(user_dataframes.get('cards', pd.DataFrame()))
        
        st.sidebar.metric("📊 Transactions Loaded", f"{transaction_count:,}")
        st.sidebar.metric("💳 Cards Loaded", f"{card_count}")
        
        # Memory optimization indicator
        if transaction_count > 0:
            st.sidebar.success("🚀 Memory Optimized: Loading only user-specific data")
        else:
            st.sidebar.warning("⚠️ No transaction data found for this user")
    
    # Feature toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Advanced Settings")
    
    # Export options
    if st.sidebar.button("📤 Export All Analysis"):
        st.sidebar.info("Export functionality would download comprehensive analysis reports")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Help section
    with st.sidebar.expander("❓ Help & Tips"):
        st.markdown("""
        **Navigation Tips:**
        • Start with Overview for basic insights
        • Use Spending Analysis for detailed breakdowns
        • Try Merchant Intelligence for subscription insights
        • Check Rewards Optimization for card recommendations
        • Explore Advanced Intelligence for AI-powered insights
        
        **Performance:**
        • Data loads only for selected user
        • All analysis runs locally and privately
        • Charts are interactive - hover and click to explore
        """)
    
    # Footer with version info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("*🔒 Privacy-First Analytics*")
    with col2:
        st.markdown("*⚡ Memory Optimized*")
    with col3:
        st.markdown("*🚀 Advanced Intelligence Built-In*")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>
            Dashboard built with Streamlit • PySpark • Advanced Analytics<br>
            Your financial data stays private and secure
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()