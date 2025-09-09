#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self, plots_dir="plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set matplotlib parameters for better looking plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def save_plot(self, filename, title="", dpi=300):
        """Save plot with consistent formatting."""
        if title:
            plt.suptitle(title, fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        else:
            plt.tight_layout()
        
        filepath = self.plots_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        return str(filepath)
    
    def plot_top_pipelines(self, df, top_n=10):
        """Create bar chart of top pipelines by volume."""
        pipeline_volumes = df.groupby('pipeline_name')['scheduled_quantity'].sum().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(pipeline_volumes)), pipeline_volumes.values / 1e9, 
                      color=sns.color_palette("viridis", len(pipeline_volumes)))
        
        plt.xlabel('Pipeline Companies')
        plt.ylabel('Total Volume (Billion Cubic Feet)')
        plt.title(f'Top {top_n} Pipeline Companies by Total Gas Volume')
        
        # Rotate labels for better readability
        pipeline_names = [name[:30] + '...' if len(name) > 30 else name for name in pipeline_volumes.index]
        plt.xticks(range(len(pipeline_volumes)), pipeline_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}B', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        return self.save_plot('top_pipelines_volume', f'Top {top_n} Pipeline Companies by Volume')
    
    def plot_state_distribution(self, df, top_n=15):
        """Create horizontal bar chart of state distribution."""
        state_counts = df['state_abb'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(state_counts)), state_counts.values,
                       color=sns.color_palette("Set2", len(state_counts)))
        
        plt.xlabel('Number of Transactions')
        plt.ylabel('States')
        plt.title(f'Top {top_n} States by Pipeline Transaction Count')
        
        plt.yticks(range(len(state_counts)), state_counts.index)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        return self.save_plot('state_distribution', f'Top {top_n} States by Transaction Count')
    
    def plot_monthly_trends(self, df):
        """Create line plot of monthly volume trends."""
        # Create monthly aggregation
        df_copy = df.copy()
        # Ensure eff_gas_day is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['eff_gas_day']):
            df_copy['eff_gas_day'] = pd.to_datetime(df_copy['eff_gas_day'])
        df_copy['year_month'] = df_copy['eff_gas_day'].dt.to_period('M')
        monthly_data = df_copy.groupby('year_month').agg({
            'scheduled_quantity': ['sum', 'mean', 'count']
        }).reset_index()
        
        monthly_data.columns = ['year_month', 'total_volume', 'avg_volume', 'transaction_count']
        monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot 1: Total Volume
        axes[0].plot(monthly_data['year_month_str'], monthly_data['total_volume'] / 1e9, 
                    marker='o', linewidth=2, markersize=4, color='steelblue')
        axes[0].set_title('Monthly Total Gas Volume Trends')
        axes[0].set_ylabel('Total Volume (Billion Cubic Feet)')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Transaction Count
        axes[1].plot(monthly_data['year_month_str'], monthly_data['transaction_count'] / 1e6,
                    marker='s', linewidth=2, markersize=4, color='darkgreen')
        axes[1].set_title('Monthly Transaction Count Trends')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Transaction Count (Millions)')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        return self.save_plot('monthly_trends', 'Monthly Gas Pipeline Trends')
    
    def plot_seasonal_patterns(self, df):
        """Create seasonal pattern analysis plots."""
        df_copy = df.copy()
        # Ensure eff_gas_day is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['eff_gas_day']):
            df_copy['eff_gas_day'] = pd.to_datetime(df_copy['eff_gas_day'])
        df_copy['month'] = df_copy['eff_gas_day'].dt.month
        df_copy['day_of_week'] = df_copy['eff_gas_day'].dt.dayofweek
        df_copy['quarter'] = df_copy['eff_gas_day'].dt.quarter
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly patterns
        monthly_avg = df_copy.groupby('month')['scheduled_quantity'].mean() / 1e6
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[0,0].bar(monthly_avg.index, monthly_avg.values, 
                     color=sns.color_palette("coolwarm", 12))
        axes[0,0].set_title('Average Volume by Month')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Average Volume (Million Cubic Feet)')
        axes[0,0].set_xticks(range(1, 13))
        axes[0,0].set_xticklabels(month_names)
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Day of week patterns
        dow_avg = df_copy.groupby('day_of_week')['scheduled_quantity'].mean() / 1e6
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        axes[0,1].bar(dow_avg.index, dow_avg.values,
                     color=sns.color_palette("viridis", 7))
        axes[0,1].set_title('Average Volume by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Average Volume (Million Cubic Feet)')
        axes[0,1].set_xticks(range(7))
        axes[0,1].set_xticklabels(dow_names)
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Quarterly patterns
        quarterly_avg = df_copy.groupby('quarter')['scheduled_quantity'].mean() / 1e6
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        
        axes[1,0].bar(quarterly_avg.index, quarterly_avg.values,
                     color=sns.color_palette("Set1", 4))
        axes[1,0].set_title('Average Volume by Quarter')
        axes[1,0].set_xlabel('Quarter')
        axes[1,0].set_ylabel('Average Volume (Million Cubic Feet)')
        axes[1,0].set_xticks(range(1, 5))
        axes[1,0].set_xticklabels(quarter_names)
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # Receipt vs Delivery patterns
        rec_del_monthly = df_copy.groupby(['month', 'rec_del_sign'])['scheduled_quantity'].sum().unstack(fill_value=0)
        rec_del_monthly.columns = ['Receipt', 'Delivery']
        
        axes[1,1].plot(rec_del_monthly.index, rec_del_monthly['Receipt'] / 1e9, 
                      marker='o', label='Receipt', linewidth=2)
        axes[1,1].plot(rec_del_monthly.index, rec_del_monthly['Delivery'] / 1e9,
                      marker='s', label='Delivery', linewidth=2)
        axes[1,1].set_title('Receipt vs Delivery by Month')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Volume (Billion Cubic Feet)')
        axes[1,1].set_xticks(range(1, 13))
        axes[1,1].set_xticklabels(month_names)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        return self.save_plot('seasonal_patterns', 'Seasonal Patterns in Gas Pipeline Operations')
    
    def plot_category_distribution(self, df):
        """Create pie chart of category distribution."""
        all_category_counts = df['category_short'].value_counts()
        
        # Group small categories (< 2% of total) into "Others"
        total_count = all_category_counts.sum()
        threshold = total_count * 0.02  # 2% threshold
        
        large_categories = all_category_counts[all_category_counts >= threshold]
        small_categories = all_category_counts[all_category_counts < threshold]
        
        # Create final categories with "Others" if needed
        if len(small_categories) > 0:
            category_counts = large_categories.copy()
            category_counts['Others'] = small_categories.sum()
        else:
            category_counts = large_categories.head(10)  # Limit to top 10 even if all are large
        
        plt.figure(figsize=(14, 10))
        colors = sns.color_palette("Set3", len(category_counts))
        
        # Create pie chart with improved label handling
        def autopct_format(pct):
            return f'{pct:.1f}%' if pct > 1.5 else ''  # Only show percentage if > 1.5%
        
        wedges, texts, autotexts = plt.pie(
            category_counts.values, 
            labels=None,  # Remove direct labels to avoid overlap
            autopct=autopct_format,
            colors=colors, 
            startangle=90,
            pctdistance=0.85,  # Move percentages closer to center
            labeldistance=1.1   # Move labels further out
        )
        
        # Improve text readability for percentages
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Create a legend instead of direct labels
        legend_labels = []
        for cat, count in zip(category_counts.index, category_counts.values):
            pct = (count / total_count) * 100
            if cat == 'Others':
                legend_labels.append(f'{cat} ({len(small_categories)} categories, {count:,} total)')
            else:
                legend_labels.append(f'{cat} ({count:,}, {pct:.1f}%)')
        
        plt.legend(wedges, legend_labels,
                  title="Pipeline Categories",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9)
        
        plt.title('Distribution of Pipeline Categories', fontsize=16, pad=20)
        
        # Ensure the pie chart is circular
        plt.axis('equal')
        
        return self.save_plot('category_distribution', 'Pipeline Category Distribution')
    
    def plot_volume_distribution(self, df, sample_size=100000):
        """Create histogram of volume distribution."""
        # Sample data for performance
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        volumes = sample_df['scheduled_quantity'].dropna()
        
        # Create two plots: one for all data, one excluding zero/very small values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: All data (99th percentile and below)
        q99 = volumes.quantile(0.99)
        volumes_filtered = volumes[volumes <= q99]
        
        ax1.hist(volumes_filtered, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Scheduled Quantity (Cubic Feet)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('All Volume Distribution (99th percentile and below)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics for all data
        mean_vol = volumes_filtered.mean()
        median_vol = volumes_filtered.median()
        ax1.axvline(mean_vol, color='red', linestyle='--', label=f'Mean: {mean_vol:,.0f}')
        ax1.axvline(median_vol, color='orange', linestyle='--', label=f'Median: {median_vol:.0f}')
        ax1.legend()
        
        # Add text box with key statistics
        stats_text = f'Total Transactions: {len(volumes):,}\n'
        stats_text += f'Zero Volume: {(volumes == 0).sum():,} ({(volumes == 0).mean()*100:.1f}%)\n'
        stats_text += f'Volume ≤ 10: {(volumes <= 10).sum():,} ({(volumes <= 10).mean()*100:.1f}%)'
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Excluding very small volumes (> 10 cubic feet)
        volumes_substantial = volumes[volumes > 10]
        if len(volumes_substantial) > 0:
            q99_substantial = volumes_substantial.quantile(0.99)
            volumes_substantial_filtered = volumes_substantial[volumes_substantial <= q99_substantial]
            
            ax2.hist(volumes_substantial_filtered, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Scheduled Quantity (Cubic Feet)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Substantial Volumes Only (> 10 cubic feet)')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics for substantial volumes
            mean_substantial = volumes_substantial_filtered.mean()
            median_substantial = volumes_substantial_filtered.median()
            ax2.axvline(mean_substantial, color='red', linestyle='--', 
                       label=f'Mean: {mean_substantial:,.0f}')
            ax2.axvline(median_substantial, color='orange', linestyle='--', 
                       label=f'Median: {median_substantial:,.0f}')
            ax2.legend()
            
            # Add text box for substantial volumes
            substantial_stats = f'Substantial Transactions: {len(volumes_substantial):,}\n'
            substantial_stats += f'({(len(volumes_substantial)/len(volumes)*100):.1f}% of total)\n'
            substantial_stats += f'Min: {volumes_substantial.min():,.0f}\n'
            substantial_stats += f'Max: {volumes_substantial.max():,.0f}'
            ax2.text(0.98, 0.98, substantial_stats, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No substantial volumes found', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        return self.save_plot('volume_distribution', 'Gas Volume Distribution Analysis')
    
    def plot_anomaly_detection(self, df, sample_size=10000):
        """Create scatter plot highlighting anomalies."""
        # Ensure eff_gas_day is datetime
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['eff_gas_day']):
            df_copy['eff_gas_day'] = pd.to_datetime(df_copy['eff_gas_day'])
        
        # Daily volume analysis
        daily_volumes = df_copy.groupby('eff_gas_day')['scheduled_quantity'].sum().reset_index()
        
        # Calculate z-scores for anomaly detection
        daily_volumes['z_score'] = np.abs((daily_volumes['scheduled_quantity'] - daily_volumes['scheduled_quantity'].mean()) / daily_volumes['scheduled_quantity'].std())
        daily_volumes['is_anomaly'] = daily_volumes['z_score'] > 3
        
        plt.figure(figsize=(16, 8))
        
        # Plot normal points
        normal_data = daily_volumes[~daily_volumes['is_anomaly']]
        anomaly_data = daily_volumes[daily_volumes['is_anomaly']]
        
        plt.scatter(normal_data['eff_gas_day'], normal_data['scheduled_quantity'] / 1e9,
                   alpha=0.6, s=20, color='blue', label='Normal')
        
        if len(anomaly_data) > 0:
            plt.scatter(anomaly_data['eff_gas_day'], anomaly_data['scheduled_quantity'] / 1e9,
                       alpha=0.8, s=60, color='red', marker='x', label='Anomalies')
        
        plt.xlabel('Date')
        plt.ylabel('Daily Volume (Billion Cubic Feet)')
        plt.title('Daily Gas Volumes with Anomaly Detection (Z-score > 3)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        return self.save_plot('anomaly_detection', 'Daily Volume Anomaly Detection')
    
    def plot_correlation_heatmap(self, df, sample_size=50000):
        """Create correlation heatmap of numeric variables."""
        # Sample data and select numeric columns
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Ensure eff_gas_day is datetime
        if not pd.api.types.is_datetime64_any_dtype(sample_df['eff_gas_day']):
            sample_df = sample_df.copy()
            sample_df['eff_gas_day'] = pd.to_datetime(sample_df['eff_gas_day'])
        
        # Create additional numeric features
        numeric_df = pd.DataFrame({
            'scheduled_quantity': sample_df['scheduled_quantity'],
            'rec_del_sign': sample_df['rec_del_sign'],
            'month': sample_df['eff_gas_day'].dt.month,
            'day_of_week': sample_df['eff_gas_day'].dt.dayofweek,
            'quarter': sample_df['eff_gas_day'].dt.quarter,
            'year': sample_df['eff_gas_day'].dt.year
        }).dropna()
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix of Pipeline Variables')
        
        return self.save_plot('correlation_heatmap', 'Variable Correlation Analysis')
    
    def create_comprehensive_dashboard(self, df):
        """Create a comprehensive dashboard with multiple visualizations."""
        print("🎨 Creating comprehensive visualization dashboard...")
        
        plots_created = []
        
        try:
            # 1. Top pipelines
            print("   📊 Creating top pipelines chart...")
            plots_created.append(self.plot_top_pipelines(df))
            
            # 2. State distribution
            print("   🗺️ Creating state distribution chart...")
            plots_created.append(self.plot_state_distribution(df))
            
            # 3. Monthly trends
            print("   📈 Creating monthly trends chart...")
            plots_created.append(self.plot_monthly_trends(df))
            
            # 4. Seasonal patterns
            print("   🔄 Creating seasonal patterns chart...")
            plots_created.append(self.plot_seasonal_patterns(df))
            
            # 5. Category distribution
            print("   🥧 Creating category pie chart...")
            plots_created.append(self.plot_category_distribution(df))
            
            # 6. Volume distribution
            print("   📊 Creating volume distribution histogram...")
            plots_created.append(self.plot_volume_distribution(df))
            
            # 7. Anomaly detection
            print("   🚨 Creating anomaly detection chart...")
            plots_created.append(self.plot_anomaly_detection(df))
            
            # 8. Correlation heatmap
            print("   🔥 Creating correlation heatmap...")
            plots_created.append(self.plot_correlation_heatmap(df))
            
            print(f"✅ Successfully created {len(plots_created)} visualizations!")
            print(f"📁 All plots saved in: {self.plots_dir}")
            
            return plots_created
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return plots_created

if __name__ == "__main__":
    # Test the visualizer
    from data_loader import DataLoader
    
    print("🎨 Testing Data Visualizer...")
    loader = DataLoader()
    df = loader.load_dataset_from_drive()
    
    if df is not None:
        visualizer = DataVisualizer()
        plots = visualizer.create_comprehensive_dashboard(df)
        print(f"\n🎉 Created {len(plots)} visualizations successfully!")
    else:
        print("❌ Failed to load dataset for visualization testing") 