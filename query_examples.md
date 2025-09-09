# Data Agent Query Examples

Quick reference for testing different types of queries with the data agent.

## 📊 Basic Dataset Information

```bash
# Get column names and data types
python data_agent.py --query "What are the 13 column names in the dataset?"

# Dataset overview
python data_agent.py --query "Give me an overview of the dataset structure"

# Basic statistics
python data_agent.py --query "Show me basic statistics about the dataset"
```

## 🏭 Pipeline Analysis

```bash
# Top pipelines by volume
python data_agent.py --query "What are the top 10 pipelines by total gas volume?"

# Pipeline operations analysis
python data_agent.py --query "Which pipelines handle the most transactions?"

# Pipeline efficiency comparison
python data_agent.py --query "Compare pipeline efficiency across different operators"
```

## 🗺️ Geographic Analysis

```bash
# State-level analysis
python data_agent.py --query "Which states have the most pipeline activity?"

# Regional patterns
python data_agent.py --query "Show me geographic patterns in gas pipeline distribution"

# County-level insights
python data_agent.py --query "What are the top counties by gas volume?"
```

## 📈 Time Series Analysis

```bash
# Seasonal patterns
python data_agent.py --query "Find seasonal patterns in natural gas pipeline operations"

# Monthly trends
python data_agent.py --query "Show me monthly trends in gas volumes over time"

# Daily patterns
python data_agent.py --query "Analyze daily patterns in gas pipeline usage"
```

## 🔍 Pattern Detection

```bash
# Operational patterns
python data_agent.py --query "Find distinct patterns in pipeline operations using machine learning"

# Location clustering
python data_agent.py --query "Cluster similar pipeline locations based on their characteristics"

# Usage patterns
python data_agent.py --query "Identify different types of gas usage patterns"
```

## 🚨 Anomaly Detection

```bash
# Volume anomalies
python data_agent.py --query "Detect unusual spikes or drops in daily gas volumes"

# Pipeline anomalies
python data_agent.py --query "Find pipelines with unusual operational patterns"

# Temporal anomalies
python data_agent.py --query "Identify unusual patterns in gas delivery timing"
```

## 🔗 Causal Analysis

```bash
# Enhanced causal analysis with confounder detection
python data_agent.py --query "What factors cause high gas volumes and what confounders should I consider?"

# Robustness testing
python data_agent.py --query "Analyze causal relationships between pipeline characteristics and volumes with robustness checks"

# Volume correlations with pathway analysis
python data_agent.py --query "What factors are correlated with high gas pipeline utilization?"

# Geographic correlations
python data_agent.py --query "How do geographic factors influence pipeline operations?"

# Temporal correlations with confounders
python data_agent.py --query "What seasonal factors affect gas pipeline demand and what confounds these relationships?"
```

## 💡 Business Intelligence Queries

```bash
# Market analysis
python data_agent.py --query "Which regions show the highest growth in gas pipeline activity?"

# Efficiency analysis
python data_agent.py --query "Identify the most and least efficient pipeline operations"

# Capacity analysis
python data_agent.py --query "Analyze pipeline capacity utilization patterns"

# Risk analysis
python data_agent.py --query "Identify potential operational risks in the pipeline network"
```

## 🎯 Complex Multi-Part Queries

```bash
# Comprehensive analysis
python data_agent.py --query "Analyze Texas pipeline operations including volume trends, seasonal patterns, and top operators"

# Comparative analysis
python data_agent.py --query "Compare pipeline operations between Texas and Louisiana including volumes, efficiency, and patterns"

# Predictive insights
python data_agent.py --query "Based on historical patterns, what insights can you provide about future gas pipeline demand?"
```

## 🚀 Interactive Mode

For extended conversations and follow-up questions:

```bash
python data_agent.py
```

Then ask questions like:
- "What did you find interesting about the data?"
- "Can you explain the seasonal patterns in more detail?"
- "What would you recommend for optimizing pipeline operations?"
- "Are there any data quality issues I should be aware of?"

## 💡 Pro Tips

1. **Be Specific**: More specific queries often yield better insights
2. **Ask Follow-ups**: Use interactive mode for deeper analysis
3. **Combine Topics**: Ask about multiple aspects in one query
4. **Business Context**: Frame questions in business terms for more relevant insights
5. **Data Quality**: Ask about data limitations and caveats for better interpretation 