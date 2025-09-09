#!/usr/bin/env python3

"""
Script to generate comprehensive visualizations for the pipeline dataset.
This creates all charts and saves them in the plots/ folder.
"""

from data_loader import DataLoader
from visualization import DataVisualizer
import time

def main():
    print("🎨 PIPELINE DATA VISUALIZATION GENERATOR")
    print("=" * 50)
    
    # Load dataset
    print("📊 Loading dataset...")
    start_time = time.time()
    
    loader = DataLoader()
    df = loader.load_dataset_from_drive()
    
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    load_time = time.time() - start_time
    print(f"✅ Dataset loaded in {load_time:.1f} seconds")
    print(f"   📈 {len(df):,} rows, {len(df.columns)} columns")
    
    # Initialize visualizer
    visualizer = DataVisualizer()
    
    # Create all visualizations
    print(f"\n🎨 Creating comprehensive visualization dashboard...")
    viz_start = time.time()
    
    plots_created = visualizer.create_comprehensive_dashboard(df)
    
    viz_time = time.time() - viz_start
    
    # Summary
    print(f"\n🎉 VISUALIZATION COMPLETE!")
    print(f"   📊 Created {len(plots_created)} visualizations")
    print(f"   ⏱️  Total time: {viz_time:.1f} seconds")
    print(f"   📁 Location: ./plots/")
    
    print(f"\n📋 Generated Plots:")
    for i, plot_path in enumerate(plots_created, 1):
        plot_name = plot_path.split('/')[-1].replace('.png', '').replace('_', ' ').title()
        print(f"   {i}. {plot_name}")
    
    print(f"\n💡 Usage Tips:")
    print(f"   • View plots: open plots/[filename].png")
    print(f"   • Use in presentations or reports")
    print(f"   • Plots are high-resolution (300 DPI)")
    print(f"   • All plots use consistent styling")

if __name__ == "__main__":
    main() 