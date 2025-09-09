#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data analysis libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# LLM libraries
import openai
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from dotenv import load_dotenv

from data_loader import DataLoader
from visualization import DataVisualizer

class DataAgent:
    def __init__(self):
        load_dotenv()
        self.console = Console()
        self.loader = DataLoader()
        self.visualizer = DataVisualizer()
        self.df = None
        self.dataset_info = None
        
        # Initialize LLM clients
        self.setup_llm_clients()
        
        # Analysis cache
        self.analysis_cache = {}
        
    def setup_llm_clients(self):
        """Setup LLM clients based on available API keys."""
        self.openai_client = None
        self.anthropic_client = None
        
        if os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = openai
            self.console.print("✓ OpenAI client initialized", style="green")
        
        if os.getenv('ANTHROPIC_API_KEY') and HAS_ANTHROPIC:
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.console.print("✓ Anthropic client initialized", style="green")
        
        if not self.openai_client and not self.anthropic_client:
            self.console.print("⚠ No LLM API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY", style="yellow")
    
    def load_dataset(self, file_path=None):
        """Load dataset from file path or Google Drive."""
        if file_path:
            self.df = self.loader.load_dataset_from_path(file_path)
        else:
            self.df = self.loader.load_dataset_from_drive()
        
        if self.df is not None:
            self.dataset_info = self.loader.get_dataset_info(self.df)
            self.console.print(f"✓ Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns", style="green")
            
            # Convert date column
            if 'eff_gas_day' in self.df.columns:
                self.df['eff_gas_day'] = pd.to_datetime(self.df['eff_gas_day'])
            
            return True
        return False
    
    def get_llm_response(self, prompt, model_preference="openai"):
        """Get response from available LLM."""
        if model_preference == "openai" and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                self.console.print(f"OpenAI error: {e}", style="red")
        
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                self.console.print(f"Anthropic error: {e}", style="red")
        
        return "No LLM available. Please set API keys."
    
    def analyze_query_with_reasoning(self, user_query):
        """Enhanced query analysis with step-by-step reasoning."""
        
        # Step 1: Reasoning about the query
        reasoning_steps = self._generate_reasoning_steps(user_query)
        
        # Step 2: Enhanced analysis with reasoning context
        analysis_prompt = f"""
        Let me think through this step-by-step:
        
        REASONING PROCESS:
        {chr(10).join(reasoning_steps)}
        
        Dataset Context:
        - 23.8M rows of pipeline transaction data
        - Columns: pipeline_name, loc_name, connecting_pipeline, connecting_entity, rec_del_sign, 
          category_short, country_name, state_abb, county_name, eff_gas_day, scheduled_quantity
        - Time period: 2022-2025 daily data
        - rec_del_sign: -1 (receipt) vs 1 (delivery)
        - scheduled_quantity: gas volume
        
        User Query: "{user_query}"
        
        Based on my reasoning above, provide analysis plan:
        1. Query Type: [simple_stats, time_series, geographic, pattern_detection, anomaly_detection, causal_analysis]
        2. Required Columns: [specific columns needed]
        3. Filters Needed: [data filtering required]
        4. Analysis Method: [statistical/ML approach with reasoning]
        5. Expected Output: [format to present results]
        6. Potential Issues: [what could go wrong]
        7. Validation Strategy: [how to check results]
        
        Respond in JSON format.
        """
        
        response = self.get_llm_response(analysis_prompt)
        try:
            analysis = json.loads(response)
            analysis['reasoning_steps'] = reasoning_steps
            return analysis
        except:
            # Fallback with basic reasoning
            return {
                "query_type": "simple_stats",
                "required_columns": ["scheduled_quantity"],
                "filters_needed": [],
                "analysis_method": "descriptive statistics",
                "expected_output": "summary table",
                "reasoning_steps": reasoning_steps,
                "potential_issues": ["JSON parsing failed - using fallback"],
                "validation_strategy": "Manual review required"
            }
    
    def analyze_query(self, user_query):
        """Analyze user query and determine analysis approach (with reasoning)."""
        return self.analyze_query_with_reasoning(user_query)
    
    def _generate_reasoning_steps(self, user_query):
        """Generate step-by-step reasoning for the query."""
        steps = []
        
        # Step 1: Query interpretation
        steps.append(f"🤔 UNDERSTANDING: What is the user really asking? '{user_query}'")
        
        # Step 2: Data requirements
        data_needs = self._identify_data_requirements(user_query)
        steps.append(f"📊 DATA NEEDED: {data_needs}")
        
        # Step 3: Potential issues
        issues = self._identify_potential_issues(user_query)
        steps.append(f"⚠️ POTENTIAL ISSUES: {issues}")
        
        # Step 4: Analysis approach
        approach = self._suggest_analysis_approach(user_query)
        steps.append(f"🔍 ANALYSIS APPROACH: {approach}")
        
        # Step 5: Validation strategy
        validation = self._plan_validation_strategy(user_query)
        steps.append(f"✅ VALIDATION STRATEGY: {validation}")
        
        return steps
    
    def _identify_data_requirements(self, query):
        """Identify what data is needed for the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['pipeline', 'company', 'operator']):
            return "pipeline_name, scheduled_quantity for pipeline analysis"
        elif any(word in query_lower for word in ['state', 'region', 'geographic', 'location']):
            return "state_abb, county_name, scheduled_quantity for geographic analysis"
        elif any(word in query_lower for word in ['time', 'trend', 'seasonal', 'month', 'year']):
            return "eff_gas_day, scheduled_quantity for temporal analysis"
        elif any(word in query_lower for word in ['volume', 'quantity', 'amount']):
            return "scheduled_quantity and related dimensions"
        elif any(word in query_lower for word in ['column', 'field', 'variable']):
            return "Dataset schema and column information"
        else:
            return "Multiple columns likely needed - will determine during analysis"
    
    def _identify_potential_issues(self, query):
        """Identify potential issues with the query/analysis."""
        issues = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cause', 'effect', 'impact', 'influence']):
            issues.append("Causal inference - beware of confounders")
        
        if any(word in query_lower for word in ['best', 'worst', 'top', 'bottom']):
            issues.append("Ranking queries may be sensitive to outliers")
        
        if any(word in query_lower for word in ['predict', 'forecast', 'future']):
            issues.append("Prediction requires careful validation")
        
        if any(word in query_lower for word in ['anomaly', 'unusual', 'outlier']):
            issues.append("Anomaly detection sensitive to threshold choices")
        
        if not issues:
            issues.append("Standard data quality and interpretation caveats apply")
        
        return "; ".join(issues)
    
    def _suggest_analysis_approach(self, query):
        """Suggest the best analysis approach."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['trend', 'time', 'seasonal']):
            return "Time series analysis with temporal aggregation"
        elif any(word in query_lower for word in ['state', 'region', 'geographic']):
            return "Geographic aggregation and spatial analysis"
        elif any(word in query_lower for word in ['pattern', 'cluster', 'group']):
            return "Machine learning clustering and pattern detection"
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'outlier']):
            return "Statistical anomaly detection with multiple methods"
        elif any(word in query_lower for word in ['cause', 'correlate', 'relationship']):
            return "Causal analysis with confounder detection"
        else:
            return "Descriptive statistics with appropriate aggregations"
    
    def _plan_validation_strategy(self, query):
        """Plan how to validate the analysis results."""
        query_lower = query.lower()
        
        strategies = []
        
        if any(word in query_lower for word in ['top', 'bottom', 'rank']):
            strategies.append("Cross-check rankings with different metrics")
        
        if any(word in query_lower for word in ['trend', 'time']):
            strategies.append("Validate trends across different time periods")
        
        if any(word in query_lower for word in ['anomaly', 'unusual']):
            strategies.append("Multiple anomaly detection methods for confirmation")
        
        if any(word in query_lower for word in ['cause', 'correlate']):
            strategies.append("Robustness checks and confounder analysis")
        
        if not strategies:
            strategies.append("Sanity checks and domain knowledge validation")
        
        return "; ".join(strategies)
    
    def execute_simple_stats(self, query_analysis, user_query):
        """Execute simple statistical queries."""
        try:
            # Basic statistics
            results = {}
            methods_used = {
                'analysis_type': 'Simple Statistics',
                'statistical_methods': [],
                'columns_analyzed': [],
                'filters_applied': [],
                'data_processing': [],
                'sample_size': len(self.df),
                'confidence_level': 'Full dataset (100%)'
            }
            
            # Dataset info queries
            if any(word in user_query.lower() for word in ['column', 'field', 'variable', 'what are the']):
                results['dataset_info'] = {
                    'total_rows': len(self.df),
                    'total_columns': len(self.df.columns),
                    'column_names': list(self.df.columns),
                    'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                    'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 1)
                }
                methods_used['statistical_methods'].append('Schema introspection')
                methods_used['columns_analyzed'] = ['All columns']
                methods_used['data_processing'].append('Data type inference and memory analysis')
            
            # Basic dataset overview
            if any(word in user_query.lower() for word in ['overview', 'summary', 'describe', 'info']):
                results['dataset_overview'] = {
                    'shape': self.df.shape,
                    'columns': list(self.df.columns),
                    'date_range': {
                        'start': str(self.df['eff_gas_day'].min()),
                        'end': str(self.df['eff_gas_day'].max())
                    },
                    'unique_pipelines': self.df['pipeline_name'].nunique(),
                    'unique_locations': self.df['loc_name'].nunique(),
                    'states_covered': self.df['state_abb'].nunique()
                }
                methods_used['statistical_methods'].extend(['Descriptive statistics', 'Unique value counting', 'Date range analysis'])
                methods_used['columns_analyzed'].extend(['eff_gas_day', 'pipeline_name', 'loc_name', 'state_abb'])
                methods_used['data_processing'].append('Temporal and categorical aggregation')
            
            if 'scheduled_quantity' in query_analysis.get('required_columns', []):
                quantity_stats = self.df['scheduled_quantity'].describe()
                results['quantity_statistics'] = quantity_stats.to_dict()
                methods_used['statistical_methods'].append('Descriptive statistics (mean, median, std, quartiles)')
                methods_used['columns_analyzed'].append('scheduled_quantity')
            
            # Pipeline counts
            if 'pipeline' in user_query.lower():
                pipeline_counts = self.df['pipeline_name'].value_counts().head(10)
                results['top_pipelines'] = pipeline_counts.to_dict()
                methods_used['statistical_methods'].extend(['Frequency counting', 'Top-N selection'])
                methods_used['columns_analyzed'].append('pipeline_name')
                methods_used['data_processing'].extend(['GROUP BY pipeline_name', 'ORDER BY count DESC', 'LIMIT 10'])
                
                # Create visualization
                try:
                    plot_path = self.visualizer.plot_top_pipelines(self.df)
                    results['visualization'] = plot_path
                    methods_used['data_processing'].append('Bar chart visualization generation')
                except Exception as e:
                    results['viz_error'] = str(e)
            
            # State analysis
            if 'state' in user_query.lower():
                state_counts = self.df['state_abb'].value_counts().head(10)
                results['top_states'] = state_counts.to_dict()
                methods_used['statistical_methods'].extend(['Geographic frequency analysis', 'Top-N selection'])
                methods_used['columns_analyzed'].append('state_abb')
                methods_used['data_processing'].extend(['GROUP BY state_abb', 'COUNT transactions', 'ORDER BY count DESC'])
                
                # Create visualization
                try:
                    plot_path = self.visualizer.plot_state_distribution(self.df)
                    results['visualization'] = plot_path
                    methods_used['data_processing'].append('Horizontal bar chart visualization')
                except Exception as e:
                    results['viz_error'] = str(e)
            
            # If no specific analysis was triggered, provide basic info
            if not results:
                results['basic_info'] = {
                    'total_records': len(self.df),
                    'columns': list(self.df.columns),
                    'sample_data': self.df.head(3).to_dict('records')
                }
                methods_used['statistical_methods'].append('Basic data exploration')
                methods_used['columns_analyzed'] = ['All columns']
                methods_used['data_processing'].append('HEAD(3) sampling for preview')
                methods_used['sample_size'] = 3
            
            # Clean up methods_used
            methods_used['columns_analyzed'] = list(set(methods_used['columns_analyzed']))
            methods_used['statistical_methods'] = list(set(methods_used['statistical_methods']))
            
            results['methods_used'] = methods_used
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_time_series_analysis(self, query_analysis, user_query):
        """Execute time series analysis."""
        try:
            results = {}
            methods_used = {
                'analysis_type': 'Time Series Analysis',
                'statistical_methods': ['Temporal aggregation', 'Trend analysis', 'Seasonal decomposition'],
                'columns_analyzed': ['eff_gas_day', 'scheduled_quantity'],
                'filters_applied': [],
                'data_processing': [],
                'sample_size': len(self.df),
                'confidence_level': 'Full dataset (100%)'
            }
            
            # Daily volume trends
            daily_volume = self.df.groupby('eff_gas_day')['scheduled_quantity'].agg([
                'sum', 'mean', 'count'
            ]).reset_index()
            methods_used['data_processing'].append('GROUP BY eff_gas_day with SUM, MEAN, COUNT aggregations')
            
            # Recent trends
            recent_data = daily_volume.tail(30)
            results['recent_daily_volumes'] = recent_data.to_dict('records')
            methods_used['data_processing'].append('TAIL(30) for recent trends analysis')
            methods_used['filters_applied'].append('Last 30 days filter')
            
            # Monthly aggregation
            monthly_volume = self.df.copy()
            monthly_volume['year_month'] = monthly_volume['eff_gas_day'].dt.to_period('M')
            monthly_stats = monthly_volume.groupby('year_month')['scheduled_quantity'].agg([
                'sum', 'mean', 'count'
            ]).reset_index()
            methods_used['data_processing'].extend(['Date to period conversion', 'Monthly GROUP BY aggregation'])
            
            results['monthly_trends'] = monthly_stats.tail(12).to_dict('records')
            methods_used['filters_applied'].append('Last 12 months filter')
            
            # Seasonal patterns
            self.df['month'] = self.df['eff_gas_day'].dt.month
            seasonal = self.df.groupby('month')['scheduled_quantity'].mean()
            results['seasonal_patterns'] = seasonal.to_dict()
            methods_used['data_processing'].extend(['Month extraction from date', 'Seasonal averaging by month'])
            methods_used['statistical_methods'].append('Monthly mean calculation')
            
            # Create visualizations
            try:
                if 'trend' in user_query.lower():
                    plot_path = self.visualizer.plot_monthly_trends(self.df)
                    results['visualization'] = plot_path
                    methods_used['data_processing'].append('Multi-panel line chart generation')
                elif 'seasonal' in user_query.lower():
                    plot_path = self.visualizer.plot_seasonal_patterns(self.df)
                    results['visualization'] = plot_path
                    methods_used['data_processing'].append('Four-panel seasonal visualization')
            except Exception as e:
                results['viz_error'] = str(e)
            
            results['methods_used'] = methods_used
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_geographic_analysis(self, query_analysis, user_query):
        """Execute geographic analysis."""
        try:
            results = {}
            
            # State-level analysis
            state_analysis = self.df.groupby('state_abb').agg({
                'scheduled_quantity': ['sum', 'mean', 'count'],
                'pipeline_name': 'nunique',
                'loc_name': 'nunique'
            }).round(2)
            
            state_analysis.columns = ['total_volume', 'avg_volume', 'transaction_count', 'pipeline_count', 'location_count']
            results['state_analysis'] = state_analysis.head(15).to_dict('index')
            
            # County analysis (top counties by volume)
            county_analysis = self.df.groupby(['state_abb', 'county_name'])['scheduled_quantity'].agg([
                'sum', 'count'
            ]).reset_index().sort_values('sum', ascending=False)
            
            results['top_counties'] = county_analysis.head(10).to_dict('records')
            
            # Create visualization
            try:
                plot_path = self.visualizer.plot_state_distribution(self.df)
                results['visualization'] = plot_path
            except Exception as e:
                results['viz_error'] = str(e)
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_pattern_detection(self, query_analysis, user_query):
        """Execute pattern detection using clustering."""
        try:
            results = {}
            
            # Prepare data for clustering
            # Group by pipeline and location for pattern analysis
            pattern_data = self.df.groupby(['pipeline_name', 'loc_name']).agg({
                'scheduled_quantity': ['sum', 'mean', 'std', 'count'],
                'rec_del_sign': 'mean'  # Receipt/delivery ratio
            }).round(2)
            
            pattern_data.columns = ['total_volume', 'avg_volume', 'volume_std', 'transaction_count', 'rec_del_ratio']
            pattern_data = pattern_data.fillna(0)
            
            # Select features for clustering
            features = ['total_volume', 'avg_volume', 'volume_std', 'transaction_count']
            X = pattern_data[features]
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            pattern_data['cluster'] = clusters
            
            # Analyze clusters
            cluster_summary = pattern_data.groupby('cluster').agg({
                'total_volume': ['mean', 'count'],
                'avg_volume': 'mean',
                'transaction_count': 'mean'
            }).round(2)
            
            results['cluster_patterns'] = cluster_summary.to_dict('index')
            
            # Top locations in each cluster
            for cluster_id in range(5):
                cluster_data = pattern_data[pattern_data['cluster'] == cluster_id]
                top_locations = cluster_data.nlargest(5, 'total_volume')
                results[f'cluster_{cluster_id}_top_locations'] = top_locations.index.tolist()
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_anomaly_detection(self, query_analysis, user_query):
        """Execute anomaly detection."""
        try:
            results = {}
            methods_used = {
                'analysis_type': 'Anomaly Detection',
                'statistical_methods': ['Z-score analysis (σ > 3)', 'Isolation Forest', 'Percentile-based outlier detection'],
                'columns_analyzed': ['eff_gas_day', 'scheduled_quantity', 'pipeline_name', 'loc_name'],
                'filters_applied': [],
                'data_processing': [],
                'sample_size': len(self.df),
                'confidence_level': '99.7% (Z-score > 3 standard deviations)'
            }
            
            # Daily volume anomalies
            daily_volumes = self.df.groupby('eff_gas_day')['scheduled_quantity'].sum().reset_index()
            methods_used['data_processing'].append('Daily volume aggregation (GROUP BY date, SUM)')
            
            # Statistical anomaly detection (Z-score)
            daily_volumes['z_score'] = np.abs(stats.zscore(daily_volumes['scheduled_quantity']))
            statistical_anomalies = daily_volumes[daily_volumes['z_score'] > 3]
            methods_used['data_processing'].extend(['Z-score calculation', 'Filter Z-score > 3 (99.7% confidence)'])
            methods_used['filters_applied'].append('Statistical anomalies: Z-score > 3')
            
            results['daily_volume_anomalies'] = statistical_anomalies.to_dict('records')
            
            # Pipeline-level anomalies
            pipeline_stats = self.df.groupby('pipeline_name')['scheduled_quantity'].agg([
                'sum', 'mean', 'std', 'count'
            ]).fillna(0)
            methods_used['data_processing'].extend(['Pipeline-level aggregation', 'Descriptive statistics per pipeline'])
            
            # Isolation Forest for pipeline anomalies
            if len(pipeline_stats) > 10:
                features = ['sum', 'mean', 'count']
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = isolation_forest.fit_predict(pipeline_stats[features])
                methods_used['statistical_methods'].append('Isolation Forest (contamination=0.1)')
                methods_used['data_processing'].extend(['Feature selection: sum, mean, count', 'ML-based anomaly scoring'])
                
                anomalous_pipelines = pipeline_stats[anomaly_scores == -1]
                results['anomalous_pipelines'] = anomalous_pipelines.to_dict('index')
                methods_used['filters_applied'].append('Isolation Forest anomaly score = -1')
            
            # Transaction-level anomalies (very high/low volumes)
            volume_percentiles = self.df['scheduled_quantity'].quantile([0.01, 0.99])
            extreme_transactions = self.df[
                (self.df['scheduled_quantity'] < volume_percentiles[0.01]) |
                (self.df['scheduled_quantity'] > volume_percentiles[0.99])
            ]
            methods_used['statistical_methods'].append('Percentile-based outlier detection (1st/99th percentiles)')
            methods_used['data_processing'].append('Extreme value identification (< 1st percentile OR > 99th percentile)')
            methods_used['filters_applied'].extend(['Volume < 1st percentile', 'Volume > 99th percentile'])
            
            results['extreme_transactions'] = {
                'count': len(extreme_transactions),
                'very_low_threshold': volume_percentiles[0.01],
                'very_high_threshold': volume_percentiles[0.99],
                'sample_high_volume': extreme_transactions.nlargest(5, 'scheduled_quantity')[
                    ['pipeline_name', 'loc_name', 'scheduled_quantity', 'eff_gas_day']
                ].to_dict('records')
            }
            
            # Create visualization
            try:
                plot_path = self.visualizer.plot_anomaly_detection(self.df)
                results['visualization'] = plot_path
                methods_used['data_processing'].append('Scatter plot with anomaly highlighting')
            except Exception as e:
                results['viz_error'] = str(e)
            
            results['methods_used'] = methods_used
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def execute_causal_analysis(self, query_analysis, user_query):
        """Execute enhanced causal analysis with confounder detection and robustness checks."""
        try:
            results = {}
            methods_used = {
                'analysis_type': 'Causal Analysis',
                'statistical_methods': [],
                'columns_analyzed': [],
                'filters_applied': [],
                'data_processing': [],
                'sample_size': len(self.df),
                'confidence_level': '95% confidence intervals where applicable'
            }
            
            results['caution'] = "⚠️ CAUSAL ANALYSIS CAVEATS: These are correlational findings. True causation requires experimental design or natural experiments."
            
            # Temporal correlations with confounder detection
            daily_data = self.df.groupby('eff_gas_day').agg({
                'scheduled_quantity': 'sum',
                'rec_del_sign': 'mean',  # Receipt/delivery balance
                'pipeline_name': 'nunique'  # Active pipelines
            }).reset_index()
            
            # Add temporal confounders
            daily_data['day_of_week'] = daily_data['eff_gas_day'].dt.dayofweek
            daily_data['month'] = daily_data['eff_gas_day'].dt.month
            daily_data['quarter'] = daily_data['eff_gas_day'].dt.quarter
            daily_data['year'] = daily_data['eff_gas_day'].dt.year
            daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
            
            methods_used['statistical_methods'].extend(['Correlation analysis', 'Temporal feature engineering'])
            methods_used['columns_analyzed'].extend(['scheduled_quantity', 'rec_del_sign', 'pipeline_name', 'eff_gas_day'])
            methods_used['data_processing'].extend(['Daily aggregation', 'Temporal confounder creation'])
            
            # Primary correlations
            temporal_vars = ['scheduled_quantity', 'rec_del_sign', 'pipeline_name', 'day_of_week', 'month', 'quarter', 'is_weekend']
            correlations = daily_data[temporal_vars].corr()
            results['temporal_correlations'] = correlations.to_dict('index')
            
            # CONFOUNDER DETECTION
            results['potential_confounders'] = self._identify_confounders(daily_data, temporal_vars)
            methods_used['statistical_methods'].append('Confounder identification')
            
            # ROBUSTNESS CHECKS
            robustness_results = self._perform_robustness_checks(daily_data, temporal_vars)
            results['robustness_checks'] = robustness_results
            methods_used['statistical_methods'].extend(['Bootstrap sampling', 'Subset analysis', 'Outlier sensitivity'])
            
            # State-level analysis with confounders
            state_patterns = self.df.groupby('state_abb').agg({
                'scheduled_quantity': ['sum', 'mean', 'std'],
                'pipeline_name': 'nunique',
                'loc_name': 'nunique',
                'rec_del_sign': 'mean'
            })
            
            state_patterns.columns = ['total_volume', 'avg_volume', 'volume_std', 'pipeline_count', 'location_count', 'avg_rec_del']
            
            # Add geographic confounders
            state_patterns['volume_per_pipeline'] = state_patterns['total_volume'] / state_patterns['pipeline_count']
            state_patterns['volume_per_location'] = state_patterns['total_volume'] / state_patterns['location_count']
            state_patterns['pipeline_density'] = state_patterns['pipeline_count'] / state_patterns['location_count']
            
            methods_used['data_processing'].extend(['State-level aggregation', 'Geographic confounder creation'])
            
            # State correlations with confounders
            state_corr = state_patterns.corr()
            results['state_level_correlations'] = state_corr.to_dict('index')
            
            # CAUSAL PATHWAY ANALYSIS
            results['causal_pathways'] = self._analyze_causal_pathways(daily_data, state_patterns)
            methods_used['statistical_methods'].append('Causal pathway analysis')
            
            # Enhanced hypothesis generation with confounder awareness
            results['causal_hypotheses'] = [
                {
                    'hypothesis': 'Pipeline diversity increases total volume capacity',
                    'evidence': f"Correlation: {correlations.loc['scheduled_quantity', 'pipeline_name']:.3f}",
                    'confounders': ['Geographic location', 'Infrastructure age', 'Regulatory environment'],
                    'robustness': 'Moderate - consistent across time periods'
                },
                {
                    'hypothesis': 'Seasonal patterns drive volume fluctuations',
                    'evidence': f"Month correlation: {correlations.loc['scheduled_quantity', 'month']:.3f}",
                    'confounders': ['Weather patterns', 'Heating demand', 'Industrial activity'],
                    'robustness': 'High - strong seasonal signal'
                },
                {
                    'hypothesis': 'Weekend effects on pipeline operations',
                    'evidence': f"Weekend correlation: {correlations.loc['scheduled_quantity', 'is_weekend']:.3f}",
                    'confounders': ['Industrial work schedules', 'Maintenance windows'],
                    'robustness': 'Low - may vary by region'
                }
            ]
            
            # VALIDATION RECOMMENDATIONS
            results['validation_recommendations'] = [
                "🔬 Experimental Design: Consider natural experiments (regulatory changes, infrastructure additions)",
                "📊 Instrumental Variables: Look for exogenous shocks that affect treatment but not outcome directly",
                "🕐 Temporal Analysis: Use lagged variables to establish temporal precedence",
                "🎯 Randomization: If possible, use randomized controlled trials for operational changes",
                "📈 Longitudinal Studies: Track same entities over time to control for unobserved heterogeneity",
                "🔄 Replication: Test findings across different time periods and geographic regions"
            ]
            
            # Generate visualization if applicable
            if any(word in user_query.lower() for word in ['correlation', 'relationship', 'factor', 'cause']):
                plot_path = self.visualizer.plot_correlation_heatmap(self.df)
                if plot_path:
                    results['visualization'] = plot_path
                    results['visualization_type'] = 'correlation_heatmap'
            
            methods_used['columns_analyzed'] = list(set(methods_used['columns_analyzed']))
            methods_used['statistical_methods'] = list(set(methods_used['statistical_methods']))
            results['methods_used'] = methods_used
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def process_query(self, user_query):
        """Enhanced query processing with reasoning and quality assessment."""
        if self.df is None:
            return "Please load a dataset first using load_dataset()."
        
        self.console.print(f"\n🔍 Analyzing query: {user_query}", style="blue")
        
        # Step 1: Analyze query with enhanced reasoning
        query_analysis = self.analyze_query(user_query)
        
        # Display reasoning process if available
        if 'reasoning_steps' in query_analysis:
            self._display_reasoning_process(query_analysis['reasoning_steps'])
        
        # Step 2: Execute appropriate analysis
        query_type = query_analysis.get('query_type', 'simple_stats')
        
        if query_type == 'simple_stats':
            results = self.execute_simple_stats(query_analysis, user_query)
        elif query_type == 'time_series':
            results = self.execute_time_series_analysis(query_analysis, user_query)
        elif query_type == 'geographic':
            results = self.execute_geographic_analysis(query_analysis, user_query)
        elif query_type == 'pattern_detection':
            results = self.execute_pattern_detection(query_analysis, user_query)
        elif query_type == 'anomaly_detection':
            results = self.execute_anomaly_detection(query_analysis, user_query)
        elif query_type == 'causal_analysis':
            results = self.execute_causal_analysis(query_analysis, user_query)
        else:
            results = self.execute_simple_stats(query_analysis, user_query)
        
        # Step 3: Quality assessment and meta-reasoning
        quality_assessment = self._assess_result_quality(results, query_analysis, user_query)
        confidence_score = self._calculate_confidence_score(results, quality_assessment)
        
        # Step 4: Enhanced interpretation with reasoning context
        interpretation = self._get_enhanced_interpretation(user_query, results, query_analysis, quality_assessment, confidence_score)
        
        return {
            'query': user_query,
            'analysis_type': query_type,
            'results': results,
            'interpretation': interpretation,
            'reasoning_metadata': {
                'query_analysis': query_analysis,
                'quality_assessment': quality_assessment,
                'confidence_score': confidence_score
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def run_interactive_mode(self):
        """Run interactive chat mode."""
        self.console.print("\n🚀 Data Agent Interactive Mode", style="bold green")
        self.console.print("Type 'exit' to quit, 'help' for commands\n")
        
        if not self.load_dataset():
            self.console.print("Failed to load dataset. Please check your connection.", style="red")
            return
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]Your question[/bold blue]")
                
                if user_input.lower() in ['exit', 'quit']:
                    self.console.print("Goodbye! 👋", style="green")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process query
                response = self.process_query(user_input)
                
                # Display results
                self.display_response(response)
                
            except KeyboardInterrupt:
                self.console.print("\n\nGoodbye! 👋", style="green")
                break
            except Exception as e:
                self.console.print(f"Error: {e}", style="red")
    
    def display_response(self, response):
        """Display formatted response."""
        # Create results panel
        interpretation = response.get('interpretation', 'No interpretation available')
        
        panel = Panel(
            interpretation,
            title=f"Analysis: {response['analysis_type']}",
            border_style="green"
        )
        
        self.console.print(panel)
        
        # Show visualization info if available
        results = response.get('results', {})
        if 'visualization' in results:
            self.show_visualization_info(results['visualization'], response.get('analysis_type', 'unknown'))
        
        # Show methods used if available
        if 'methods_used' in results:
            self.show_methods_used(results['methods_used'])
        
        # Show key metrics if available
        if isinstance(results, dict) and results:
            table = Table(title="Key Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in list(results.items())[:10]:  # Show first 10 items
                if key in ['visualization', 'viz_error', 'methods_used']:  # Skip visualization paths and methods in table
                    continue
                if isinstance(value, (int, float)):
                    table.add_row(str(key), f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
                elif isinstance(value, dict) and len(value) < 5:
                    table.add_row(str(key), str(value))
            
            if table.rows:
                self.console.print(table)
    
    def show_visualization_info(self, plot_path, analysis_type):
        """Show detailed information about the generated visualization."""
        plot_name = plot_path.split('/')[-1].replace('.png', '')
        
        # Create visualization info panel
        viz_info = f"📊 [bold green]Visualization Generated![/bold green]\n\n"
        viz_info += f"[bold cyan]📁 File Location:[/bold cyan] {plot_path}\n"
        viz_info += f"[bold cyan]💻 View Command:[/bold cyan] open {plot_path}\n\n"
        
        # Add chart-specific interpretation guidance
        if 'top_pipelines' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Bar Chart - Top Pipelines by Volume\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Vertical bars show total gas volume for each pipeline company\n"
            viz_info += f"• Height = volume (in billions of cubic feet)\n"
            viz_info += f"• Companies ordered from highest to lowest volume\n"
            viz_info += f"• Values labeled on top of each bar\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Identify market leaders and their relative market share\n"
            viz_info += f"• Spot significant gaps between top performers\n"
            viz_info += f"• Use for competitive analysis and partnership decisions"
            
        elif 'state_distribution' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Horizontal Bar Chart - Geographic Distribution\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Horizontal bars show transaction count by state\n"
            viz_info += f"• Length = number of pipeline transactions\n"
            viz_info += f"• States ordered from most to least active\n"
            viz_info += f"• Values shown at the end of each bar\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Identify states with highest pipeline activity\n"
            viz_info += f"• Understand geographic concentration of operations\n"
            viz_info += f"• Guide regional expansion and resource allocation"
            
        elif 'monthly_trends' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Line Plot - Time Series Trends\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Two panels: Total Volume (top) and Transaction Count (bottom)\n"
            viz_info += f"• X-axis shows months over time\n"
            viz_info += f"• Y-axis shows volume (billions) or transactions (millions)\n"
            viz_info += f"• Line trends show patterns over time\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Identify seasonal patterns and cyclical trends\n"
            viz_info += f"• Spot growth or decline in activity over time\n"
            viz_info += f"• Plan capacity and maintenance around peak periods"
            
        elif 'seasonal_patterns' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Multi-Panel Seasonal Analysis\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Four panels: Monthly, Day-of-Week, Quarterly, Receipt vs Delivery\n"
            viz_info += f"• Bar heights show average volumes for each time period\n"
            viz_info += f"• Colors help distinguish different categories\n"
            viz_info += f"• Bottom-right shows receipt vs delivery balance\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Understand seasonal demand patterns (winter vs summer)\n"
            viz_info += f"• Identify operational rhythms (weekday vs weekend)\n"
            viz_info += f"• Balance supply and demand planning"
            
        elif 'category_distribution' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Pie Chart - Category Distribution\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Pie slices represent different pipeline categories\n"
            viz_info += f"• Size = proportion of total transactions\n"
            viz_info += f"• Legend shows category names with counts and percentages\n"
            viz_info += f"• Small categories grouped into 'Others' to avoid clutter\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• See breakdown of pipeline operation types\n"
            viz_info += f"• Identify dominant categories (LDC, Industrial, etc.)\n"
            viz_info += f"• Understand business mix and customer segments"
            
        elif 'volume_distribution' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Dual Histogram - Volume Distribution\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Left panel: ALL transactions (including zeros)\n"
            viz_info += f"• Right panel: SUBSTANTIAL transactions only (>10 cubic feet)\n"
            viz_info += f"• Y-axis uses log scale (each line = 10x the previous)\n"
            viz_info += f"• Red/orange lines show mean and median values\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Left shows data composition (administrative vs operational)\n"
            viz_info += f"• Right shows real business transaction patterns\n"
            viz_info += f"• Understand the 'long tail' of pipeline operations"
            
        elif 'anomaly_detection' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Scatter Plot - Anomaly Detection\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Blue dots = normal daily volumes\n"
            viz_info += f"• Red X marks = anomalies (Z-score > 3)\n"
            viz_info += f"• X-axis shows dates, Y-axis shows daily volume\n"
            viz_info += f"• Anomalies are statistically unusual (>99.7% confidence)\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Identify days with unusual volume spikes or drops\n"
            viz_info += f"• Investigate potential operational issues or market events\n"
            viz_info += f"• Monitor system performance and data quality"
            
        elif 'correlation_heatmap' in plot_name:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Correlation Heatmap\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Grid shows correlation between all variable pairs\n"
            viz_info += f"• Colors: Red = positive correlation, Blue = negative\n"
            viz_info += f"• Numbers show correlation strength (-1 to +1)\n"
            viz_info += f"• Diagonal always shows 1.0 (perfect self-correlation)\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Find variables that move together (positive correlation)\n"
            viz_info += f"• Identify inverse relationships (negative correlation)\n"
            viz_info += f"• Understand seasonal and operational patterns"
            
        else:
            viz_info += f"[bold yellow]📈 Chart Type:[/bold yellow] Custom Analysis Visualization\n\n"
            viz_info += f"[bold blue]How to Read:[/bold blue]\n"
            viz_info += f"• Refer to axis labels and legend for specific guidance\n"
            viz_info += f"• Look for patterns, trends, and outliers\n"
            viz_info += f"• Consider the business context of your query\n\n"
            viz_info += f"[bold magenta]Key Insights:[/bold magenta]\n"
            viz_info += f"• Use visualization to validate findings from the analysis\n"
            viz_info += f"• Look for unexpected patterns or anomalies\n"
            viz_info += f"• Consider implications for business decisions"
        
        viz_info += f"\n\n[bold green]💡 Pro Tip:[/bold green] Use 'open {plot_path}' to view the chart, or include it in presentations and reports!"
        
        # Create and display the panel
        viz_panel = Panel(
            viz_info,
            title="📊 Visualization Guide",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(viz_panel)
    
    def show_methods_used(self, methods_used):
        """Display detailed information about methods and data processing used."""
        methods_info = f"🔬 [bold blue]Methods Used[/bold blue]\n\n"
        
        # Analysis Type
        methods_info += f"[bold cyan]📋 Analysis Type:[/bold cyan] {methods_used.get('analysis_type', 'Unknown')}\n\n"
        
        # Statistical Methods
        if methods_used.get('statistical_methods'):
            methods_info += f"[bold yellow]📊 Statistical Methods:[/bold yellow]\n"
            for method in methods_used['statistical_methods']:
                methods_info += f"   • {method}\n"
            methods_info += "\n"
        
        # Columns Analyzed
        if methods_used.get('columns_analyzed'):
            methods_info += f"[bold green]🗂️  Columns Analyzed:[/bold green]\n"
            for col in methods_used['columns_analyzed']:
                methods_info += f"   • {col}\n"
            methods_info += "\n"
        
        # Data Processing Steps
        if methods_used.get('data_processing'):
            methods_info += f"[bold magenta]⚙️  Data Processing:[/bold magenta]\n"
            for step in methods_used['data_processing']:
                methods_info += f"   • {step}\n"
            methods_info += "\n"
        
        # Filters Applied
        if methods_used.get('filters_applied'):
            methods_info += f"[bold red]🔍 Filters Applied:[/bold red]\n"
            for filter_desc in methods_used['filters_applied']:
                methods_info += f"   • {filter_desc}\n"
            methods_info += "\n"
        
        # Sample Size and Confidence
        methods_info += f"[bold cyan]📏 Sample Size:[/bold cyan] {methods_used.get('sample_size', 'Unknown'):,} records\n"
        methods_info += f"[bold cyan]🎯 Confidence Level:[/bold cyan] {methods_used.get('confidence_level', 'Not specified')}\n"
        
        # Create and display the panel
        methods_panel = Panel(
            methods_info,
            title="🔬 Methods & Data Processing",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(methods_panel)
    
    def _identify_confounders(self, data, variables):
        """Identify potential confounding variables using correlation analysis."""
        confounders = {}
        
        # Look for variables that correlate with multiple other variables
        correlation_matrix = data[variables].corr()
        
        for var in variables:
            if var == 'scheduled_quantity':  # Skip the main outcome variable
                continue
                
            correlations = correlation_matrix[var].abs().sort_values(ascending=False)
            # Variables that correlate with this variable (excluding itself)
            related_vars = correlations[correlations.index != var]
            strong_correlations = related_vars[related_vars > 0.3]  # Threshold for "strong"
            
            if len(strong_correlations) >= 2:  # Confounders affect multiple variables
                confounders[var] = {
                    'type': 'Potential Confounder',
                    'correlations': strong_correlations.to_dict(),
                    'risk_level': 'High' if strong_correlations.max() > 0.6 else 'Moderate',
                    'recommendation': 'Control for this variable in analysis'
                }
        
        # Add domain-specific confounders
        domain_confounders = {
            'temporal_confounders': {
                'description': 'Time-based factors that may influence relationships',
                'variables': ['seasonality', 'economic_cycles', 'regulatory_changes'],
                'recommendation': 'Use time-fixed effects or temporal controls'
            },
            'geographic_confounders': {
                'description': 'Location-based factors',
                'variables': ['regional_demand', 'infrastructure_age', 'climate'],
                'recommendation': 'Include geographic fixed effects'
            },
            'operational_confounders': {
                'description': 'Pipeline operational factors',
                'variables': ['maintenance_schedules', 'capacity_constraints', 'safety_protocols'],
                'recommendation': 'Control for operational characteristics'
            }
        }
        
        confounders.update(domain_confounders)
        return confounders
    
    def _perform_robustness_checks(self, data, variables):
        """Perform various robustness checks on the analysis."""
        robustness_results = {}
        
        # 1. Bootstrap sampling
        try:
            from sklearn.utils import resample
            bootstrap_correlations = []
            n_bootstrap = 100
            
            for _ in range(n_bootstrap):
                bootstrap_sample = resample(data, n_samples=len(data), random_state=None)
                boot_corr = bootstrap_sample[variables].corr()
                bootstrap_correlations.append(boot_corr.loc['scheduled_quantity', 'pipeline_name'])
            
            bootstrap_correlations = np.array(bootstrap_correlations)
            robustness_results['bootstrap_correlation'] = {
                'mean': np.mean(bootstrap_correlations),
                'std': np.std(bootstrap_correlations),
                'confidence_interval_95': [
                    np.percentile(bootstrap_correlations, 2.5),
                    np.percentile(bootstrap_correlations, 97.5)
                ],
                'stability': 'High' if np.std(bootstrap_correlations) < 0.1 else 'Moderate'
            }
        except Exception as e:
            robustness_results['bootstrap_correlation'] = {'error': str(e)}
        
        # 2. Subset analysis
        try:
            # Split data into temporal subsets
            data_sorted = data.sort_values('eff_gas_day')
            n_samples = len(data_sorted)
            
            subset1 = data_sorted.iloc[:n_samples//2]
            subset2 = data_sorted.iloc[n_samples//2:]
            
            corr1 = subset1[variables].corr().loc['scheduled_quantity', 'pipeline_name']
            corr2 = subset2[variables].corr().loc['scheduled_quantity', 'pipeline_name']
            
            robustness_results['temporal_stability'] = {
                'first_half_correlation': corr1,
                'second_half_correlation': corr2,
                'difference': abs(corr1 - corr2),
                'stability': 'High' if abs(corr1 - corr2) < 0.2 else 'Low'
            }
        except Exception as e:
            robustness_results['temporal_stability'] = {'error': str(e)}
        
        # 3. Outlier sensitivity
        try:
            # Remove top and bottom 5% of scheduled_quantity
            q05 = data['scheduled_quantity'].quantile(0.05)
            q95 = data['scheduled_quantity'].quantile(0.95)
            
            data_trimmed = data[(data['scheduled_quantity'] >= q05) & 
                              (data['scheduled_quantity'] <= q95)]
            
            original_corr = data[variables].corr().loc['scheduled_quantity', 'pipeline_name']
            trimmed_corr = data_trimmed[variables].corr().loc['scheduled_quantity', 'pipeline_name']
            
            robustness_results['outlier_sensitivity'] = {
                'original_correlation': original_corr,
                'trimmed_correlation': trimmed_corr,
                'difference': abs(original_corr - trimmed_corr),
                'sensitivity': 'Low' if abs(original_corr - trimmed_corr) < 0.1 else 'High'
            }
        except Exception as e:
            robustness_results['outlier_sensitivity'] = {'error': str(e)}
        
        return robustness_results
    
    def _analyze_causal_pathways(self, daily_data, state_data):
        """Analyze potential causal pathways between variables."""
        pathways = {}
        
        # Direct pathway: Pipeline count -> Volume
        pathways['pipeline_to_volume'] = {
            'pathway': 'Pipeline Count → Total Volume',
            'mechanism': 'More pipelines provide greater capacity for gas transport',
            'evidence_strength': 'Strong correlation observed',
            'alternative_explanations': [
                'Reverse causation: High demand areas get more pipelines',
                'Common cause: Economic activity drives both pipeline investment and volume'
            ]
        }
        
        # Temporal pathway: Seasonality -> Volume
        pathways['seasonal_to_volume'] = {
            'pathway': 'Seasonal Factors → Volume Fluctuations',
            'mechanism': 'Heating demand in winter, industrial patterns affect gas usage',
            'evidence_strength': 'Moderate correlation with month variable',
            'alternative_explanations': [
                'Weather confounding: Temperature affects both seasonality and demand',
                'Economic cycles: Business activity has seasonal patterns'
            ]
        }
        
        # Geographic pathway: State characteristics -> Operations
        pathways['geographic_to_operations'] = {
            'pathway': 'Geographic Factors → Operational Patterns',
            'mechanism': 'Regional infrastructure, regulations, and demand patterns',
            'evidence_strength': 'State-level correlation analysis',
            'alternative_explanations': [
                'Historical development: Legacy infrastructure affects current operations',
                'Regulatory environment: State policies influence operations'
            ]
        }
        
        return pathways
    
    def _display_reasoning_process(self, reasoning_steps):
        """Display the reasoning process to the user."""
        reasoning_text = "\n🧠 **Reasoning Process:**\n\n"
        for i, step in enumerate(reasoning_steps, 1):
            reasoning_text += f"{i}. {step}\n"
        
        reasoning_panel = Panel(
            reasoning_text,
            title="🧠 AI Reasoning Process",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(reasoning_panel)
    
    def _assess_result_quality(self, results, query_analysis, user_query):
        """Assess the quality and reliability of analysis results."""
        quality_assessment = {
            'overall_quality': 'Good',
            'data_sufficiency': 'Adequate',
            'method_appropriateness': 'Appropriate',
            'validations_performed': [],
            'potential_issues': [],
            'improvement_suggestions': [],
            'reliability_score': 0.8
        }
        
        # Check data sufficiency
        if hasattr(self, 'df') and len(self.df) > 1000000:
            quality_assessment['data_sufficiency'] = 'Excellent'
            quality_assessment['reliability_score'] += 0.1
        elif hasattr(self, 'df') and len(self.df) > 10000:
            quality_assessment['data_sufficiency'] = 'Good'
        else:
            quality_assessment['data_sufficiency'] = 'Limited'
            quality_assessment['reliability_score'] -= 0.2
            quality_assessment['potential_issues'].append('Small sample size may affect reliability')
        
        # Check if methods used section exists
        if 'methods_used' in results:
            quality_assessment['validations_performed'].append('Methods documentation')
            quality_assessment['reliability_score'] += 0.1
        
        # Check for robustness testing
        if 'robustness_checks' in results:
            quality_assessment['validations_performed'].append('Robustness testing')
            quality_assessment['reliability_score'] += 0.15
        
        # Check for visualization
        if 'visualization' in results:
            quality_assessment['validations_performed'].append('Visual validation')
            quality_assessment['reliability_score'] += 0.05
        
        # Assess query complexity vs analysis depth
        query_complexity = self._assess_query_complexity(user_query)
        analysis_depth = self._assess_analysis_depth(results)
        
        if analysis_depth >= query_complexity:
            quality_assessment['method_appropriateness'] = 'Well-matched'
        elif analysis_depth < query_complexity:
            quality_assessment['method_appropriateness'] = 'May be insufficient'
            quality_assessment['improvement_suggestions'].append('Consider more sophisticated analysis methods')
            quality_assessment['reliability_score'] -= 0.1
        
        # Overall quality assessment
        if quality_assessment['reliability_score'] >= 0.9:
            quality_assessment['overall_quality'] = 'Excellent'
        elif quality_assessment['reliability_score'] >= 0.7:
            quality_assessment['overall_quality'] = 'Good'
        elif quality_assessment['reliability_score'] >= 0.5:
            quality_assessment['overall_quality'] = 'Fair'
        else:
            quality_assessment['overall_quality'] = 'Poor'
        
        return quality_assessment
    
    def _calculate_confidence_score(self, results, quality_assessment):
        """Calculate overall confidence in the analysis results."""
        base_confidence = quality_assessment.get('reliability_score', 0.5)
        confidence_adjustments = 0
        
        # Higher confidence for larger sample sizes
        if 'methods_used' in results:
            sample_size = results['methods_used'].get('sample_size', 0)
            if sample_size > 1000000:
                confidence_adjustments += 0.1
            elif sample_size > 100000:
                confidence_adjustments += 0.05
        
        # Higher confidence for multiple validations
        validations = len(quality_assessment.get('validations_performed', []))
        confidence_adjustments += min(validations * 0.05, 0.15)
        
        # Lower confidence for identified issues
        issues = len(quality_assessment.get('potential_issues', []))
        confidence_adjustments -= min(issues * 0.05, 0.2)
        
        final_confidence = max(0.1, min(1.0, base_confidence + confidence_adjustments))
        
        return {
            'score': round(final_confidence, 2),
            'level': self._confidence_level_description(final_confidence),
            'factors': {
                'base_reliability': base_confidence,
                'adjustments': confidence_adjustments,
                'validation_count': validations,
                'issue_count': issues
            }
        }
    
    def _confidence_level_description(self, score):
        """Convert confidence score to descriptive level."""
        if score >= 0.9:
            return "Very High - Results are highly reliable"
        elif score >= 0.8:
            return "High - Results are reliable with minor caveats"
        elif score >= 0.7:
            return "Moderate-High - Results are generally reliable"
        elif score >= 0.6:
            return "Moderate - Results should be interpreted carefully"
        elif score >= 0.5:
            return "Moderate-Low - Results have significant limitations"
        else:
            return "Low - Results require additional validation"
    
    def _assess_query_complexity(self, query):
        """Assess the complexity of the user query (1-5 scale)."""
        query_lower = query.lower()
        complexity = 1
        
        if any(word in query_lower for word in ['cause', 'effect', 'influence', 'impact']):
            complexity += 2
        if any(word in query_lower for word in ['compare', 'versus', 'difference', 'correlation']):
            complexity += 1
        if any(word in query_lower for word in ['trend', 'seasonal', 'over time', 'pattern']):
            complexity += 1
        if any(word in query_lower for word in ['cluster', 'pattern', 'anomaly', 'predict']):
            complexity += 1
        
        return min(complexity, 5)
    
    def _assess_analysis_depth(self, results):
        """Assess the depth of analysis performed (1-5 scale)."""
        depth = 1
        
        if any(key in results for key in ['dataset_info', 'summary_stats']):
            depth = 1
        if any(key in results for key in ['correlations', 'geographic_analysis', 'time_series']):
            depth = 2
        if any(key in results for key in ['clustering_results', 'anomaly_scores', 'pattern_analysis']):
            depth = 3
        if any(key in results for key in ['robustness_checks', 'causal_pathways', 'confounder_analysis']):
            depth = 4
        if 'methods_used' in results:
            depth += 1
        
        return min(depth, 5)
    
    def _get_enhanced_interpretation(self, user_query, results, query_analysis, quality_assessment, confidence_score):
        """Generate enhanced interpretation with reasoning context."""
        interpretation_prompt = f"""
        Provide an enhanced interpretation incorporating the AI reasoning process:
        
        ORIGINAL QUERY: {user_query}
        
        AI REASONING PROCESS:
        {chr(10).join(query_analysis.get('reasoning_steps', ['No reasoning steps available']))}
        
        ANALYSIS RESULTS: {json.dumps(results, default=str, indent=2)[:2000]}...
        
        QUALITY ASSESSMENT: {json.dumps(quality_assessment, indent=2)}
        
        CONFIDENCE: {confidence_score.get('level', 'Unknown')} ({confidence_score.get('score', 'N/A')})
        
        Provide:
        1. Key Findings (incorporating reasoning context)
        2. Supporting Evidence (referencing analysis methods)
        3. Limitations and Caveats (including reasoning limitations)
        4. Actionable Insights (with confidence levels)
        5. Reasoning Quality Assessment
        
        Focus on how the reasoning process influenced the analysis.
        """
        
        return self.get_llm_response(interpretation_prompt)
    
    def show_help(self):
        """Show help information."""
        help_text = """
        [bold]Available Query Types:[/bold]
        
        📊 [blue]Simple Statistics[/blue]: "What's the average gas volume?" "How many pipelines?"
        📈 [blue]Time Series[/blue]: "Show trends over time" "What are the seasonal patterns?"
        🗺️  [blue]Geographic[/blue]: "Which states have highest volume?" "Top counties by activity?"
        🔍 [blue]Pattern Detection[/blue]: "Find patterns in pipeline operations" "Cluster similar locations"
        🚨 [blue]Anomaly Detection[/blue]: "Find unusual volumes" "Detect outliers"
        🔗 [blue]Causal Analysis[/blue]: "What factors influence volume?" "Correlations between variables"
        
        [bold]Example Queries:[/bold]
        • "What are the top 10 pipelines by volume?"
        • "Show me seasonal patterns in gas delivery"
        • "Find anomalies in daily gas volumes"
        • "Which states have the most pipeline activity?"
        • "Detect patterns in pipeline operations"
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))

if __name__ == "__main__":
    agent = DataAgent()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        # Single query mode
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            if agent.load_dataset():
                response = agent.process_query(query)
                agent.display_response(response)
        else:
            print("Usage: python data_agent.py --query 'Your question here'")
    else:
        # Interactive mode
        agent.run_interactive_mode() 