# Data Agent Project

A sophisticated chat-based data analysis agent that can interact with large datasets and answer complex analytical questions using natural language queries. Built for analyzing natural gas pipeline data with advanced pattern recognition, anomaly detection, and causal analysis capabilities.

## 🚀 Features

- **Natural Language Interface**: Ask questions in plain English
- **Comprehensive Analysis Types**:
  - Simple statistics and aggregations
  - Time series analysis and seasonal patterns
  - Geographic analysis and regional insights
  - Pattern detection using machine learning clustering
  - Anomaly detection with multiple algorithms
  - Causal analysis with correlation insights
- **LLM Integration**: Supports both OpenAI GPT-4 and Anthropic Claude
- **Large Dataset Handling**: Efficiently processes 23M+ row datasets
- **Interactive CLI**: User-friendly command-line interface
- **Automated Data Loading**: Downloads and processes data from Google Drive
- **Rich Visualizations**: Automatically generates charts, graphs, and plots
- **Professional Charts**: High-resolution plots with consistent styling

## 📊 Dataset

The agent works with natural gas pipeline data containing:
- **23.8 million records** of pipeline transactions
- **Daily data** from 2022-2025
- **169 pipeline companies** across the United States
- **Geographic coverage** with state and county information
- **Transaction details** including volumes and receipt/delivery indicators

**Note**: The dataset is automatically downloaded from Google Drive and is not included in this repository.

## 🛠 Installation & Quick Start

### Prerequisites
- Python 3.10+
- Conda environment (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd data-agent-project
```

2. **Create conda environment** (if using conda):
```bash
conda create -n data-agent python=3.12
conda activate data-agent
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up API keys** (choose one or both):
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Running the Agent

**Interactive Mode** (recommended):
```bash
python data_agent.py
```

**Single Query Mode**:
```bash
python data_agent.py --query "What are the top 10 pipelines by volume?"
```

**Generate All Visualizations**:
```bash
python create_plots.py
```

## 📝 Example Queries & Outputs

### Simple Statistics
**Query**: "What are the top 10 pipelines by total gas volume?"

**Output**: 
```
Key Findings:
• Texas Eastern Transmission leads with 2.1B cubic feet total volume
• Top 10 pipelines handle 65% of all gas transactions
• Interstate pipelines dominate the highest volume categories
• Average transaction size varies significantly by pipeline operator
```

### Time Series Analysis
**Query**: "Show me seasonal patterns in gas delivery over the past year"

**Output**:
```
Key Findings:
• Winter months (Dec-Feb) show 40% higher volumes than summer
• Peak demand occurs in January with average 15M cf/day
• Weekend volumes are typically 12% lower than weekdays
• Strong correlation (0.78) between temperature drops and volume spikes
```

### Geographic Analysis
**Query**: "Which states have the most pipeline activity?"

**Output**:
```
Key Findings:
• Texas leads with 3.2M transactions across 45 counties
• Louisiana has highest volume density (volume per square mile)
• Interstate corridors show concentrated pipeline activity
• 5 states account for 60% of all natural gas pipeline traffic
```

### Anomaly Detection
**Query**: "Find unusual patterns in daily gas volumes"

**Output**:
```
Key Findings:
• Detected 23 days with volumes >3 standard deviations from normal
• February 15, 2023 shows highest anomaly (250% above average)
• 3 pipelines consistently show irregular volume patterns
• Most anomalies correlate with extreme weather events
• **Visualization**: Scatter plot showing anomalies highlighted in red
```

### Pattern Recognition
**Query**: "Find clusters of similar pipeline operations"

**Output**:
```
Key Findings:
• Identified 5 distinct operational patterns:
  - High-volume interstate corridors (15% of locations)
  - Regional distribution hubs (25% of locations)  
  - Local delivery points (45% of locations)
  - Storage facilities (10% of locations)
  - Interconnection points (5% of locations)
```

## 🔧 Dataset Loading Options

### Option 1: Automatic Download (Default)
The agent automatically downloads the dataset from Google Drive:
```python
agent = DataAgent()
agent.load_dataset()  # Downloads from Google Drive
```

### Option 2: Local File Path
If you have the dataset locally:
```python
agent = DataAgent()
agent.load_dataset("/path/to/your/dataset.parquet")
```

### Option 3: Environment Variable
Set the dataset path via environment variable:
```bash
export DATASET_PATH="/path/to/dataset.parquet"
python data_agent.py
```

## 🧠 Analysis Capabilities

### Statistical Analysis
- Descriptive statistics and distributions
- Correlation analysis and hypothesis testing
- Time series decomposition and trend analysis

### Machine Learning
- K-means clustering for pattern detection
- Isolation Forest for anomaly detection
- Principal Component Analysis for dimensionality reduction

### Geographic Analysis
- State and county-level aggregations
- Regional pattern identification
- Geographic correlation analysis

### Causal Inference
- Correlation analysis with caveats
- Temporal relationship identification
- Confounding variable detection

## 📊 Visualization Features

The agent automatically generates professional visualizations for analysis results:

### Chart Types Available
- **Bar Charts**: Top pipelines, state distributions
- **Line Plots**: Monthly trends, time series analysis
- **Pie Charts**: Category distributions
- **Histograms**: Volume distributions with statistical overlays
- **Scatter Plots**: Anomaly detection with highlighted outliers
- **Heatmaps**: Correlation matrices
- **Multi-panel Plots**: Seasonal patterns (monthly, weekly, quarterly)

### Visualization Capabilities
- **Automatic Generation**: Charts created based on query type
- **High Resolution**: 300 DPI for presentations and reports
- **Professional Styling**: Consistent colors, fonts, and layouts
- **Smart Sampling**: Efficient processing of large datasets
- **Multiple Formats**: PNG format with transparent backgrounds

### Generated Visualizations
When you run queries, the agent automatically creates relevant charts:
- Pipeline volume queries → Bar charts of top pipelines
- Geographic queries → State distribution charts
- Time series queries → Line plots and seasonal patterns
- Anomaly detection → Scatter plots with outliers highlighted
- Pattern analysis → Multi-panel comparison charts

## ⚠️ Assumptions & Limitations

### Data Quality
- **Missing Coordinates**: Latitude/longitude data is 100% missing
- **Incomplete Connections**: 78% of connecting_pipeline data is missing
- **Data Validation**: Assumes scheduled_quantity represents actual volumes

### Analysis Limitations
- **Causation vs Correlation**: Causal analysis identifies correlations, not true causation
- **External Factors**: Weather, economic, and regulatory factors not included
- **Temporal Scope**: Analysis limited to 2022-2025 data period
- **Geographic Precision**: Analysis limited to state/county level due to missing coordinates

### Performance Considerations
- **Memory Usage**: Full dataset requires ~13GB RAM
- **Query Speed**: Complex analyses may take 10-30 seconds
- **LLM Dependency**: Interpretation quality depends on API availability

### Methodological Caveats
- Clustering results may vary with different parameters
- Anomaly detection thresholds are statistically derived
- Seasonal patterns assume consistent operational practices
- Geographic analysis limited by data completeness

## 🏗 Architecture

```
data_agent.py          # Main agent with LLM integration
├── DataAgent          # Core analysis engine
├── LLM Integration    # OpenAI/Anthropic clients
├── Analysis Modules   # Specialized analysis functions
└── Interactive CLI    # User interface

data_loader.py         # Dataset loading and processing
├── DataLoader         # Multi-format data loading
├── Google Drive API   # Remote dataset access
└── Data Validation    # Schema inference and cleaning

explore_data.py        # Dataset exploration utilities
requirements.txt       # Python dependencies
```

## 🔍 Technical Details

### Supported File Formats
- Parquet (recommended for large datasets)
- CSV/TSV files
- Excel files (.xlsx, .xls)

### Analysis Algorithms
- **Clustering**: K-means, DBSCAN
- **Anomaly Detection**: Isolation Forest, Statistical Z-score
- **Time Series**: Seasonal decomposition, trend analysis
- **Statistics**: Pearson correlation, descriptive statistics

### LLM Integration
- **Query Understanding**: Natural language to analysis mapping
- **Result Interpretation**: Technical results to business insights
- **Context Awareness**: Dataset-specific domain knowledge

## 🤝 Contributing

This project was built for SynMax evaluation. Key design decisions:

1. **Modularity**: Separate concerns for data loading, analysis, and presentation
2. **Scalability**: Efficient handling of large datasets with chunking and sampling
3. **Extensibility**: Easy to add new analysis types and data sources
4. **User Experience**: Natural language interface with rich formatted output

## 📄 License

This project is created for evaluation purposes. All dependencies maintain their respective licenses.

---

**Built with**: Python 3.12, pandas, scikit-learn, OpenAI GPT-4, Anthropic Claude, Rich CLI 