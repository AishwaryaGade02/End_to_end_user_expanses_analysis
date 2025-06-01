import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.rewards_analysis import RewardsOptimizer

def display_optimization_overview(optimization):
    """FIXED: Enhanced optimization overview with proper pandas handling"""
    st.subheader("🎯 Portfolio Optimization Results")
    
    results = optimization.get('optimization_results', {})
    optimal_portfolio = optimization.get('optimal_portfolio', {})
    
    # ==== MAIN METRICS SECTION ====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        improvement = results.get('annual_improvement', 0)
        improvement_pct = results.get('improvement_percentage', 0)
        
        # Dynamic color based on improvement
        if improvement > 100:
            delta_color = "normal"
            emoji = "🎉"
        elif improvement > 50:
            delta_color = "normal" 
            emoji = "💡"
        else:
            delta_color = "off"
            emoji = "⚖️"
        
        st.metric(
            f"{emoji} Annual Improvement",
            f"${improvement:.2f}",
            delta=f"{improvement_pct:.1f}% boost",
            delta_color=delta_color
        )
    
    with col2:
        signup_bonuses = results.get('signup_bonuses', 0)
        st.metric(
            "🎁 Welcome Bonuses",
            f"${signup_bonuses:.2f}",
            delta="One-time earnings",
            help="Total signup bonuses from recommended cards"
        )
    
    with col3:
        payback_period = results.get('payback_period', 0)
        if payback_period > 0:
            payback_months = int(payback_period * 12)
            payback_text = f"{payback_months} months"
            payback_delta = "to break even"
            payback_color = "normal" if payback_months <= 12 else "inverse"
        else:
            payback_text = "Immediate"
            payback_delta = "No fees to recover"
            payback_color = "normal"
        
        st.metric(
            "⏱️ Payback Period",
            payback_text,
            delta=payback_delta,
            delta_color=payback_color
        )
    
    
    # ==== DYNAMIC RECOMMENDATION CALLOUT ====
    st.markdown("---")
    
    if improvement > 100:
        st.success(f"🎉 **Excellent Opportunity!** Optimizing could earn you **${improvement:.0f} more per year** — that's like getting a {improvement/12:.0f}% monthly bonus on your spending!")
    elif improvement > 50:
        st.info(f"💡 **Good Potential** — ${improvement:.0f} additional annual rewards possible with the right card strategy.")
    elif improvement > 0:
        st.warning(f"⚖️ **Minor Gains Available** — ${improvement:.0f} yearly improvement. Consider if the effort is worth it.")
    else:
        st.success("✅ **Already Optimized!** Your current strategy is working well.")
    
    # ==== PORTFOLIO INSIGHT CARDS ====
    if optimal_portfolio:
        col1, col2 = st.columns(2)
        
        with col1:
            total_fees = optimal_portfolio.get('total_annual_fees', 0)
            recommended_cards = len(optimal_portfolio.get('cards', []))
            
            if total_fees > 0:
                st.info(f"💳 **{recommended_cards}-Card Strategy** with ${total_fees:.0f} total annual fees")
            else:
                st.success(f"🆓 **{recommended_cards}-Card Strategy** with zero annual fees")
        
        with col2:
            # Best category insight
            spending_analysis = optimization.get('spending_analysis', {})
            if spending_analysis and spending_analysis.get('annual_spending_by_category'):
                top_category = max(spending_analysis['annual_spending_by_category'].items(), key=lambda x: x[1])
                st.info(f"🏆 **Top Category**: {top_category[0]} (${top_category[1]:,.0f}/year)")

    # ==== COMPLETE REWARDS ANALYSIS SECTION ====
    st.markdown("---")
    st.subheader("📊 Category-by-Category Rewards Analysis")
    
    # Add storytelling intro
    spending_analysis = optimization.get('spending_analysis', {})
    if not spending_analysis or not spending_analysis.get('annual_spending_by_category'):
        st.warning("⚠️ No detailed spending breakdown available for category analysis.")
        return
    
    # Brief analysis summary first
    total_categories = len(spending_analysis['annual_spending_by_category'])
    total_spending = spending_analysis.get('total_annual_spending', 0)
    
    st.markdown(f"""
    📈 **Analysis Summary**: Analyzing **{total_categories} spending categories** across **${total_spending:,.0f}** in annual spending.
    The optimization below shows where you can earn the most additional rewards.
    """)
    
    # Calculate detailed rewards by category
    spending_by_category = spending_analysis['annual_spending_by_category']
    
    # Create detailed analysis
    detailed_rewards_analysis = []
    
    for category, spending in spending_by_category.items():
        # Current rewards (assume 1% default)
        current_rewards = spending * 0.01
        
        # Find best potential rate from optimal portfolio
        best_rate = 0.01  # Default
        best_card = "Current Card"
        
        if optimal_portfolio and optimal_portfolio.get('card_details'):
            for card_detail in optimal_portfolio['card_details']:
                rate = card_detail['categories'].get(category, card_detail['categories']['default'])
                if rate > best_rate:
                    best_rate = rate
                    best_card = card_detail['name']
        
        potential_rewards = spending * best_rate
        additional_rewards = potential_rewards - current_rewards
        
        detailed_rewards_analysis.append({
            'category': category,
            'spending': spending,
            'current_rewards': current_rewards,
            'potential_rewards': potential_rewards,
            'additional_rewards': additional_rewards,
            'best_card': best_card,
            'best_reward_rate': best_rate * 100,
            'improvement_pct': (additional_rewards / current_rewards * 100) if current_rewards > 0 else 0
        })
    
    rewards_df = pd.DataFrame(detailed_rewards_analysis)
    rewards_df = rewards_df.sort_values('additional_rewards', ascending=False)
    
    if not rewards_df.empty:
        # Overview metrics for rewards analysis
        total_current = rewards_df['current_rewards'].sum()
        total_potential = rewards_df['potential_rewards'].sum()
        total_additional = rewards_df['additional_rewards'].sum()
        
        # Show only the most important metric to avoid redundancy
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎯 Optimization Potential",
                f"${total_additional:.2f}",
                f"+{(total_additional/total_current*100):.0f}% vs current" if total_current > 0 else "N/A"
            )
        
        with col2:
            # Show best opportunity - FIXED: Check if dataframe has data before accessing
            if len(rewards_df) > 0:
                best_opportunity = rewards_df.iloc[0]
                if best_opportunity['additional_rewards'] > 0:
                    category_name = best_opportunity['category']
                    display_name = category_name[:20] + "..." if len(category_name) > 20 else category_name
                    st.metric(
                        "🏆 Best Category",
                        display_name,
                        f"+${best_opportunity['additional_rewards']:.2f}/year"
                    )
        
        with col3:
            # Show recommended strategy
            if optimal_portfolio and optimal_portfolio.get('card_details'):
                primary_card = optimal_portfolio['card_details'][0]['name']
                display_name = primary_card[:20] + "..." if len(primary_card) > 20 else primary_card
                st.metric(
                    "💳 Primary Recommendation", 
                    display_name,
                    "Best overall value"
                )
        
        # Interactive visualizations
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enhanced comparison chart with annotations
            top_categories = rewards_df.head(8).copy()
            
            fig_comparison = go.Figure()
            
            # Current rewards
            fig_comparison.add_trace(go.Bar(
                name='Current Rewards',
                x=top_categories['category'],
                y=top_categories['current_rewards'],
                marker_color='lightblue',
                text=top_categories['current_rewards'].round(2),
                texttemplate='$%{text}',
                textposition='inside'
            ))
            
            # Potential rewards
            fig_comparison.add_trace(go.Bar(
                name='Optimized Rewards',
                x=top_categories['category'],
                y=top_categories['potential_rewards'],
                marker_color='darkgreen',
                text=top_categories['potential_rewards'].round(2),
                texttemplate='$%{text}',
                textposition='inside'
            ))
            
            # Add annotations for biggest opportunities - FIXED: Check if data exists
            if len(top_categories) > 0:
                biggest_opportunity = top_categories.iloc[0]
                fig_comparison.add_annotation(
                    x=biggest_opportunity['category'],
                    y=biggest_opportunity['potential_rewards'],
                    text=f"↗️ Best opportunity: +${biggest_opportunity['additional_rewards']:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="yellow",
                    bordercolor="red"
                )
            
            fig_comparison.update_layout(
                title="Current vs Optimized Rewards by Category",
                xaxis_title="Spending Category",
                yaxis_title="Annual Rewards ($)",
                barmode='group',
                height=400,
                xaxis_tickangle=-45,
                showlegend=True
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # Opportunity matrix - spending vs potential gain
            fig_matrix = px.scatter(
                rewards_df.head(10),
                x='spending',
                y='additional_rewards',
                size='improvement_pct',
                hover_name='category',
                title="Opportunity Matrix",
                labels={
                    'spending': 'Annual Spending ($)',
                    'additional_rewards': 'Additional Rewards ($)',
                    'improvement_pct': 'Improvement %'
                },
                color='additional_rewards',
                color_continuous_scale='viridis'
            )
            
            # Add quadrant lines - FIXED: Use scalar values instead of Series
            avg_spending = float(rewards_df['spending'].median())
            avg_additional = float(rewards_df['additional_rewards'].median())
            
            fig_matrix.add_hline(y=avg_additional, line_dash="dash", line_color="gray", annotation_text="Avg Opportunity")
            fig_matrix.add_vline(x=avg_spending, line_dash="dash", line_color="gray", annotation_text="Avg Spending")
            
            fig_matrix.update_layout(height=400)
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Smart insights section - only show actionable ones - FIXED: Use proper boolean indexing
        significant_opportunities = rewards_df[rewards_df['additional_rewards'] > 5.0]
        
        if len(significant_opportunities) > 0:  # FIXED: Use len() instead of direct boolean
            st.subheader("🎯 Your Top 3 Action Items")
            
            for i, (_, opportunity) in enumerate(significant_opportunities.head(3).iterrows(), 1):
                with st.container():
                    # Create action-oriented cards
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. Switch {opportunity['category']} spending to {opportunity['best_card']}**")
                        st.caption(f"Current: 1% → Optimized: {opportunity['best_reward_rate']:.1f}% rewards")
                    
                    with col2:
                        st.metric("💰 Extra Earnings", f"${opportunity['additional_rewards']:.0f}/year")
                    
                    with col3:
                        # Calculate monthly impact
                        monthly_impact = opportunity['additional_rewards'] / 12
                        st.metric("📅 Monthly Impact", f"${monthly_impact:.0f}")
                    
                    if i < 3:  # Don't add separator after last item
                        st.markdown("---")
        else:
            st.success("✅ **Great Job!** You're already maximizing rewards across all major categories.")
        
        # Collapsible detailed table
        with st.expander("📋 View Complete Category Analysis", expanded=False):
            # Enhanced table with better formatting
            display_df = rewards_df.copy()
            display_df['category_short'] = display_df['category'].apply(lambda x: x[:25] + "..." if len(x) > 25 else x)
            display_df['best_card_short'] = display_df['best_card'].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)
            
            st.dataframe(
                display_df[['category_short', 'spending', 'current_rewards', 'potential_rewards', 'additional_rewards', 'best_card_short', 'best_reward_rate']],
                column_config={
                    "category_short": "Category",
                    "spending": st.column_config.NumberColumn("Annual Spending", format="$%,.0f"),
                    "current_rewards": st.column_config.NumberColumn("Current Rewards", format="$%.2f"),
                    "potential_rewards": st.column_config.NumberColumn("Optimized Rewards", format="$%.2f"),
                    "additional_rewards": st.column_config.NumberColumn("Additional Rewards", format="$%.2f"),
                    "best_card_short": "Best Card",
                    "best_reward_rate": st.column_config.NumberColumn("Reward Rate", format="%.1f%%")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            col1, col2 = st.columns(2)
            
            with col1:
                csv_rewards = rewards_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Full Analysis",
                    data=csv_rewards,
                    file_name=f"rewards_optimization_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create action plan export - FIXED: Check if significant_opportunities exists
                if len(significant_opportunities) > 0:
                    action_plan = []
                    for i, (_, opp) in enumerate(significant_opportunities.head(3).iterrows(), 1):
                        action_plan.append({
                            'Priority': i,
                            'Action': f"Use {opp['best_card']} for {opp['category']}",
                            'Current_Rate': '1%',
                            'New_Rate': f"{opp['best_reward_rate']:.1f}%",
                            'Annual_Gain': f"${opp['additional_rewards']:.2f}"
                        })
                    
                    if action_plan:
                        action_df = pd.DataFrame(action_plan)
                        csv_actions = action_df.to_csv(index=False)
                        st.download_button(
                            label="📋 Download Action Plan",
                            data=csv_actions,
                            file_name=f"rewards_action_plan.csv",
                            mime="text/csv"
                        )
    else:
        st.warning("⚠️ Unable to generate category analysis due to insufficient spending data.")
    
    # Quick tip at the bottom
    st.markdown("---")
    st.info("💡 **Pro Tip**: Focus on your top 2-3 spending categories first. Small changes in high-spending areas often yield the biggest rewards boost!")


def display_card_comparison_table(optimization, rewards_optimizer):
    """Enhanced card comparison with better context and storytelling"""
    st.subheader("💰 Credit Card Performance for Your Spending")
    
    spending_analysis = optimization.get('spending_analysis')
    optimal_portfolio = optimization.get('optimal_portfolio')
    
    if not spending_analysis:
        st.warning("Unable to generate card comparison due to insufficient spending data.")
        return
    
    # Add context about why cards are ranked this way
    total_spending = spending_analysis.get('total_annual_spending', 0)
    st.markdown(f"""
    📊 **Analysis Context**: Rankings based on your **${total_spending:,.0f}** annual spending pattern. 
    Single cards are ranked by *total net value*, while the *optimal portfolio* considers *category coverage*.
    """)
    
    # Generate comparison table
    comparison_table = rewards_optimizer.generate_card_comparison_table(spending_analysis)
    
    if not comparison_table.empty:
        # Add ranking and context columns
        comparison_table['Rank'] = range(1, len(comparison_table) + 1)
        
        # Add a "Best For" column based on card strengths
        comparison_table['Best_For'] = comparison_table['Card'].map({
            'Chase Freedom Unlimited': 'General spending + Restaurants',
            'Chase Sapphire Preferred': 'Travel + Dining',
            'American Express Gold': 'Heavy restaurant/grocery spending',
            'Citi Double Cash': 'Everything (2% flat rate)',
            'Capital One Savor': 'Dining + Entertainment',
            'Discover it Cash Back': 'Rotating categories + First year'
        })
        
        # Reorder and format columns
        display_columns = ['Rank', 'Card', 'Net Rewards', 'Best_For', 'Annual Fee', 'Signup Bonus', 'First Year Value']
        comparison_table = comparison_table[display_columns]
        
        # Color-code the dataframe for better UX
        st.dataframe(
            comparison_table,
            column_config={
                "Rank": st.column_config.NumberColumn("📈 Rank", format="%d", width="small"),
                "Card": st.column_config.TextColumn("💳 Card Name", width="medium"),
                "Net Rewards": st.column_config.NumberColumn("💰 Net Annual Value", format="$%.0f", width="medium"),
                "Best_For": st.column_config.TextColumn("🎯 Best For", width="large"),
                "Annual Fee": st.column_config.NumberColumn("💳 Annual Fee", format="$%d", width="small"),
                "Signup Bonus": st.column_config.NumberColumn("🎁 Welcome Bonus", format="$%d", width="medium"),
                "First Year Value": st.column_config.NumberColumn("🚀 First Year Total", format="$%.0f", width="medium")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Add explanation of portfolio vs single card rankings
        st.info("""
        🤔 **Why does the optimal portfolio differ from this ranking?** 
        - **This table** ranks single cards by total value
        - **The optimal portfolio** combines cards for maximum category coverage
        - A lower-ranked card might be perfect for specific spending categories
        """)
        
        # Enhanced top 3 recommendations with context
        st.subheader("🏆 Top Recommendations Explained")
        
        top_3 = comparison_table.head(3)
        
        for idx, (_, card) in enumerate(top_3.iterrows(), 1):
            medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            
            # Determine recommendation reason
            if card['Annual Fee'] == 0 and card['Net Rewards'] > 1000:
                reason = "🆓 High value with no annual fee"
            elif card['Annual Fee'] > 0 and card['Net Rewards'] > 1500:
                reason = "💪 High rewards justify the annual fee"
            elif 'Double Cash' in card['Card']:
                reason = "🎯 Simple 2% on everything"
            else:
                reason = "⚡ Strong overall performance"
            
            with st.expander(f"{medal} **#{idx}: {card['Card']}** - {reason}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annual Value", f"${card['Net Rewards']:.0f}")
                    if card['Annual Fee'] > 0:
                        payback_months = (card['Annual Fee'] / (card['Net Rewards'] + card['Annual Fee'])) * 12
                        st.caption(f"Fee pays back in ~{payback_months:.0f} months")
                    else:
                        st.caption("✅ No annual fee")
                
                with col2:
                    st.metric("Welcome Bonus", f"${card['Signup Bonus']:.0f}")
                    st.caption("One-time earning")
                
                with col3:
                    st.metric("First Year Total", f"${card['First Year Value']:.0f}")
                    roi = ((card['First Year Value'] - card['Annual Fee']) / card['Annual Fee'] * 100) if card['Annual Fee'] > 0 else 0
                    if roi > 0:
                        st.caption(f"ROI: {roi:.0f}%")
                    else:
                        st.caption("Immediate value")
                
                # Show specific strengths
                st.markdown(f"**🎯 Best Use Case**: {card['Best_For']}")
                
                # Add specific category rates if this card is in optimal portfolio
                if optimal_portfolio and card['Card'] in [detail['name'] for detail in optimal_portfolio.get('card_details', [])]:
                    st.success("⭐ **This card is in your optimal portfolio!**")
                    
                    # Show why it's recommended
                    card_detail = next((detail for detail in optimal_portfolio['card_details'] if detail['name'] == card['Card']), None)
                    if card_detail:
                        best_categories = []
                        for category, rate in card_detail['categories'].items():
                            if category != 'default' and rate >= 0.03:  # 3% or higher
                                best_categories.append(f"**{category}**: {rate*100:.0f}% rewards")
                        
                        if best_categories:
                            st.markdown("**🎯 Your High-Reward Categories:**")
                            for cat in best_categories[:3]:  # Show top 3
                                st.markdown(f"• {cat}")
        
        # Portfolio strategy explanation
        st.markdown("---")
        st.subheader("🧠 Portfolio Strategy Insights")
        
        if optimal_portfolio:
            recommended_cards = [detail['name'] for detail in optimal_portfolio.get('card_details', [])]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Your Optimal Portfolio:**")
                for i, card_name in enumerate(recommended_cards, 1):
                    # Find the rank of this card in the table
                    card_rank = comparison_table[comparison_table['Card'] == card_name]['Rank'].iloc[0] if not comparison_table[comparison_table['Card'] == card_name].empty else "N/A"
                    st.markdown(f"{i}. **{card_name}** (Ranked #{card_rank} overall)")
            
            with col2:
                st.markdown("**💡 Why This Combination?**")
                
                # Calculate coverage
                spending_by_category = spending_analysis.get('annual_spending_by_category', {})
                if spending_by_category:
                    total_coverage = 0
                    covered_spending = 0
                    
                    for category, spending in spending_by_category.items():
                        total_coverage += spending
                        best_rate = 0.01  # Default
                        
                        for card_detail in optimal_portfolio.get('card_details', []):
                            rate = card_detail['categories'].get(category, card_detail['categories']['default'])
                            best_rate = max(best_rate, rate)
                        
                        if best_rate > 0.015:  # Better than 1.5%
                            covered_spending += spending
                    
                    coverage_pct = (covered_spending / total_coverage * 100) if total_coverage > 0 else 0
                    
                    st.markdown(f"""
                    - **{coverage_pct:.0f}%** of your spending gets enhanced rewards
                    - **Category optimization** beats single-card approach
                    - **Balanced** fee vs. no-fee strategy
                    """)
        
        # Action-oriented summary
        st.markdown("---")
        best_single_card = comparison_table.iloc[0]['Card']
        best_portfolio_value = optimal_portfolio.get('net_annual_rewards', 0) if optimal_portfolio else 0
        best_single_value = comparison_table.iloc[0]['Net Rewards']
        
        if best_portfolio_value > best_single_value:
            portfolio_advantage = best_portfolio_value - best_single_value
            st.success(f"""
            🎯 **Bottom Line**: While **{best_single_card}** is the best single card (${best_single_value:.0f} value), 
            your **optimal portfolio strategy** earns **${portfolio_advantage:.0f} more** (${best_portfolio_value:.0f} total) 
            through smart category coverage!
            """)
        else:
            st.info(f"""
            🎯 **Bottom Line**: **{best_single_card}** performs so well for your spending pattern 
            that a single-card strategy might be your simplest approach.
            """)
    
    else:
        st.error("Unable to generate card comparison table.")

def display_impact_analysis(optimization):
    """Display detailed impact analysis"""
    st.subheader("📈 Optimization Impact Analysis")
    
    results = optimization.get('optimization_results', {})
    optimal_portfolio = optimization.get('optimal_portfolio')
    
    if not results:
        st.warning("No optimization results to analyze.")
        return
    
    
    # Visual impact analysis
    st.markdown("#### 📊 5-Year Projection")
    
    # Calculate 5-year projection
    years = list(range(1, 6))
    current_rewards_projection = [optimization['current_annual_rewards'] * year for year in years]
    
    if optimal_portfolio:
        optimal_rewards_projection = [optimal_portfolio['net_annual_rewards'] * year for year in years]
        
        # Add signup bonuses to first year
        signup_bonuses = results.get('signup_bonuses', 0)
        optimal_rewards_projection[0] += signup_bonuses
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=current_rewards_projection,
            mode='lines+markers',
            name='Current Portfolio',
            line=dict(color='lightblue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=optimal_rewards_projection,
            mode='lines+markers',
            name='Optimal Portfolio',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="5-Year Cumulative Rewards Projection",
            xaxis_title="Years",
            yaxis_title="Cumulative Rewards ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show total 5-year benefit
        total_5_year_benefit = optimal_rewards_projection[-1] - current_rewards_projection[-1]
        st.success(f"💰 **5-Year Total Benefit**: ${total_5_year_benefit:,.2f}")

def display_actionable_recommendations(optimization, user_dataframes):
    """Display actionable recommendations with prioritization"""
    st.subheader("🔮 Personalized Action Plan")
    
    recommendations = optimization.get('recommendations', [])
    results = optimization.get('optimization_results', {})
    optimal_portfolio = optimization.get('optimal_portfolio')
    
    if not recommendations:
        st.info("No specific recommendations available.")
        return
    
    # Priority-based recommendations
    high_priority = [r for r in recommendations if r.get('priority') == 'High']
    medium_priority = [r for r in recommendations if r.get('priority') == 'Medium']
    low_priority = [r for r in recommendations if r.get('priority') == 'Low']
    
    # High Priority Actions
    if high_priority:
        st.markdown("### 🔴 High Priority Actions")
        for i, rec in enumerate(high_priority, 1):
            with st.container():
                st.error(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Medium Priority Actions  
    if medium_priority:
        st.markdown("### 🟡 Medium Priority Actions")
        for i, rec in enumerate(medium_priority, 1):
            with st.container():
                st.warning(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Low Priority Actions
    if low_priority:
        st.markdown("### 🔵 Low Priority Actions")
        for i, rec in enumerate(low_priority, 1):
            with st.container():
                st.info(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Implementation Timeline
    st.markdown("### 📅 Suggested Implementation Timeline")
    
    if optimal_portfolio and results.get('annual_improvement', 0) > 50:
        timeline_steps = [
            "**Week 1-2**: Research and compare recommended cards",
            "**Week 3**: Apply for highest-priority card (best signup bonus)",
            "**Month 2**: Once approved, begin using new card for optimal categories",
            "**Month 3**: Apply for second card if recommended",
            "**Month 4**: Optimize spending across all cards",
            "**Month 6**: Review and track rewards earned vs projections"
        ]
        
        for step in timeline_steps:
            st.markdown(f"• {step}")
    else:
        st.info("💡 **Minor Optimization**: Consider implementing changes gradually as your current portfolio is already well-optimized.")
    
    # Quick action buttons (mockup - would integrate with real application systems)
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Email This Analysis"):
            st.success("Analysis would be emailed to you!")
    
    with col2:
        if st.button("📅 Set Reminder"):
            st.success("Reminder set to review in 3 months!")
    
    with col3:
        if st.button("💾 Save to Profile"):
            st.success("Optimization saved to your profile!")
    
    # Additional insights
    if optimal_portfolio:
        st.markdown("### 💡 Additional Insights")
        
        with st.expander("🔍 See detailed card application tips"):
            st.markdown("""
            **Credit Card Application Tips:**
            
            1. **Credit Score Impact**: Each application may temporarily lower your credit score by 5-10 points
            2. **Timing**: Space applications 2-3 months apart for best approval odds
            3. **Income Requirements**: Ensure you meet minimum income requirements
            4. **Spending Requirements**: Plan to meet signup bonus spending requirements naturally
            5. **Annual Fee Strategy**: Set calendar reminders before annual fees post
            
            **Approval Odds Factors:**
            • Current credit score
            • Income level  
            • Existing relationship with bank
            • Recent credit applications (5/24 rule for Chase)
            • Current debt levels
            """)
        
        with st.expander("📊 See spending strategy for optimal rewards"):
            spending_analysis = optimization.get('spending_analysis', {})
            if spending_analysis and spending_analysis.get('annual_spending_by_category'):
                st.markdown("**Recommended Card Usage by Category:**")
                
                # Import here to avoid circular imports
                temp_optimizer = RewardsOptimizer(
                    user_dataframes.get('cards', pd.DataFrame()),
                    user_dataframes.get('transactions', pd.DataFrame()),
                    user_dataframes.get('mcc_codes', pd.DataFrame())
                )
                
                for category, spending in spending_analysis['annual_spending_by_category'].items():
                    if spending > 100:  # Only show significant categories
                        best_card = None
                        best_rate = 0
                        
                        for card_name in optimal_portfolio['cards']:
                            card_info = temp_optimizer.card_rewards_database[card_name]
                            rate = card_info['categories'].get(category, card_info['categories']['default'])
                            if rate > best_rate:
                                best_rate = rate
                                best_card = card_name
                        
                        st.markdown(f"• **{category}** (${spending:,.0f}/year): Use {best_card} ({best_rate*100:.1f}% rewards)")

def display_rewards_optimization_page(user_dataframes, selected_user_id):
    """Enhanced rewards optimization page with all portfolio optimization features"""
    st.title("💳 Advanced Rewards & Portfolio Optimization")
    st.markdown("*Comprehensive credit card portfolio optimization with real market data and actionable insights*")
    st.markdown("---")
    
    # Initialize rewards optimizer
    rewards_optimizer = RewardsOptimizer(
        user_dataframes.get('cards', pd.DataFrame()),
        user_dataframes.get('transactions', pd.DataFrame()),
        user_dataframes.get('mcc_codes', pd.DataFrame())
    )
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Optimization Controls")
    
    # Optimization parameters
    time_period_options = {
        "Last 3 Months": "3_months",
        "Last 6 Months": "6_months", 
        "Last Year": "1_year"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=2
    )
    
    time_period = time_period_options[selected_period_display]
    
    max_cards = st.sidebar.slider(
        "🃏 Maximum Cards in Portfolio",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of cards to recommend for optimal portfolio"
    )
    
    # Advanced optimization options
    st.sidebar.subheader("⚙️ Advanced Options")
    include_annual_fees = st.sidebar.checkbox("💳 Include cards with annual fees", value=True)
    prioritize_signup_bonuses = st.sidebar.checkbox("🎁 Prioritize signup bonuses", value=True)
    show_detailed_analysis = st.sidebar.checkbox("📊 Show detailed analysis", value=True)
    
    # Get optimization results
    with st.spinner("🔄 Optimizing your credit card portfolio..."):
        optimization = rewards_optimizer.optimize_card_portfolio(selected_user_id, time_period, max_cards)
    
    if not optimization:
        st.warning("⚠️ Unable to perform portfolio optimization due to limited transaction data.")
        st.info("💡 Try selecting a different time period or ensure you have sufficient transaction history.")
        return
    
    # Main content tabs - Enhanced with more comprehensive analysis
    tab1, tab3, tab4, tab5 = st.tabs([
        "🎯 Optimization Results", 
        "💰 Card Comparison",
        "📈 Impact Analysis",
        "🔮 Recommendations"
    ])
    
    with tab1:
        display_optimization_overview(optimization)
    
    with tab3:
        display_card_comparison_table(optimization, rewards_optimizer)
    
    with tab4:
        display_impact_analysis(optimization)
    
    with tab5:
        display_actionable_recommendations(optimization, user_dataframes)