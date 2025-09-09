#!/usr/bin/env python3

from data_agent import DataAgent
import json

def test_basic_functionality():
    """Test the data agent's core functionality without LLM."""
    print("🧪 Testing Data Agent Core Functionality\n")
    
    # Initialize agent
    agent = DataAgent()
    
    # Load dataset
    print("📊 Loading dataset...")
    if not agent.load_dataset():
        print("❌ Failed to load dataset")
        return False
    
    print(f"✅ Dataset loaded: {agent.df.shape[0]:,} rows, {agent.df.shape[1]} columns\n")
    
    # Test different analysis types
    test_queries = [
        ("Simple Stats", "top pipelines by volume"),
        ("Time Series", "seasonal patterns in gas delivery"), 
        ("Geographic", "states with most pipeline activity"),
        ("Pattern Detection", "find patterns in pipeline operations"),
        ("Anomaly Detection", "unusual patterns in daily volumes"),
        ("Causal Analysis", "factors that influence gas volume")
    ]
    
    results = {}
    
    for analysis_type, query in test_queries:
        print(f"🔍 Testing {analysis_type}...")
        try:
            # Manually call analysis functions to bypass LLM requirement
            if "stats" in query.lower() or "pipeline" in query.lower():
                result = agent.execute_simple_stats({}, query)
            elif "time" in query.lower() or "seasonal" in query.lower():
                result = agent.execute_time_series_analysis({}, query)
            elif "state" in query.lower() or "geographic" in query.lower():
                result = agent.execute_geographic_analysis({}, query)
            elif "pattern" in query.lower():
                result = agent.execute_pattern_detection({}, query)
            elif "anomaly" in query.lower() or "unusual" in query.lower():
                result = agent.execute_anomaly_detection({}, query)
            elif "factor" in query.lower() or "causal" in query.lower():
                result = agent.execute_causal_analysis({}, query)
            
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ Success - Generated {len(result)} result sections")
                results[analysis_type] = result
                
                # Show a sample of results
                if isinstance(result, dict):
                    for key, value in list(result.items())[:2]:
                        if isinstance(value, dict) and len(value) < 5:
                            print(f"      {key}: {value}")
                        elif isinstance(value, (int, float)):
                            print(f"      {key}: {value:,.2f}")
                            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()
    
    # Summary
    successful_tests = len([r for r in results.values() if 'error' not in r])
    print(f"📈 Test Summary: {successful_tests}/{len(test_queries)} analysis types working")
    
    if successful_tests == len(test_queries):
        print("🎉 All core functionality is working! Ready for LLM integration.")
    else:
        print("⚠️  Some issues found. Check the errors above.")
    
    return successful_tests == len(test_queries)

def show_sample_insights():
    """Show sample insights from the dataset."""
    print("\n" + "="*60)
    print("📊 SAMPLE DATASET INSIGHTS (without LLM)")
    print("="*60)
    
    agent = DataAgent()
    if not agent.load_dataset():
        return
    
    df = agent.df
    
    # Basic stats
    print(f"\n🔢 Dataset Overview:")
    print(f"   • Total transactions: {len(df):,}")
    print(f"   • Date range: {df['eff_gas_day'].min()} to {df['eff_gas_day'].max()}")
    print(f"   • Pipeline companies: {df['pipeline_name'].nunique()}")
    print(f"   • Unique locations: {df['loc_name'].nunique()}")
    print(f"   • States covered: {df['state_abb'].nunique()}")
    
    # Top pipelines
    print(f"\n🏭 Top 5 Pipelines by Total Volume:")
    top_pipelines = df.groupby('pipeline_name')['scheduled_quantity'].sum().sort_values(ascending=False).head()
    for i, (pipeline, volume) in enumerate(top_pipelines.items(), 1):
        print(f"   {i}. {pipeline}: {volume:,.0f}")
    
    # Geographic distribution
    print(f"\n🗺️  Top 5 States by Transaction Count:")
    top_states = df['state_abb'].value_counts().head()
    for i, (state, count) in enumerate(top_states.items(), 1):
        print(f"   {i}. {state}: {count:,} transactions")
    
    # Time patterns
    print(f"\n📅 Monthly Volume Trends (Recent):")
    monthly = df.groupby(df['eff_gas_day'].dt.to_period('M'))['scheduled_quantity'].sum().tail(6)
    for month, volume in monthly.items():
        print(f"   {month}: {volume:,.0f}")
    
    print(f"\n✨ Ready to answer complex questions with LLM integration!")

if __name__ == "__main__":
    # Run basic tests
    success = test_basic_functionality()
    
    if success:
        # Show sample insights
        show_sample_insights()
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Set your API keys: export OPENAI_API_KEY='your-key'")
    print(f"   2. Run: python data_agent.py")
    print(f"   3. Ask questions like: 'What are seasonal patterns in gas delivery?'") 