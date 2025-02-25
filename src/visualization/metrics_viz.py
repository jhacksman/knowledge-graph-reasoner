"""Metrics visualization module for knowledge graph reasoner."""
from typing import List, Dict, Any, Optional, Tuple, Set
import os
import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import json
from datetime import datetime

log = logging.getLogger(__name__)


class MetricsVisualizer:
    """Visualizes metrics from knowledge graph evaluation."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        color_palette: Optional[List[str]] = None
    ):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            color_palette: Custom color palette for visualizations
        """
        self.output_dir = output_dir or "visualizations"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default color palette
        self.color_palette = color_palette or [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
    
    def visualize_metrics_over_time(
        self,
        metrics_history: List[Dict[str, Any]],
        metrics_to_plot: Optional[List[str]] = None,
        title: str = "Metrics Over Time",
        filename: Optional[str] = None,
        show_confidence_interval: bool = True,
        window_size: int = 3
    ) -> str:
        """Visualize metrics over time.
        
        Args:
            metrics_history: List of metrics at different iterations
            metrics_to_plot: List of metric names to plot (default: all numeric)
            title: Visualization title
            filename: Output filename (without extension)
            show_confidence_interval: Whether to show confidence intervals
            window_size: Window size for confidence interval calculation
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not metrics_history:
                log.warning("No metrics history provided")
                return ""
            
            # Extract iterations
            iterations = [m.get("iteration", i) for i, m in enumerate(metrics_history)]
            
            # Determine metrics to plot
            if not metrics_to_plot:
                # Find all numeric metrics
                metrics_to_plot = []
                for key in metrics_history[0].keys():
                    if key not in ["iteration", "timestamp", "error", "domain_coverage", "communities", "bridge_nodes"]:
                        value = metrics_history[0].get(key)
                        if isinstance(value, (int, float)):
                            metrics_to_plot.append(key)
            
            # Create figure
            fig = make_subplots(
                rows=len(metrics_to_plot), 
                cols=1,
                subplot_titles=[m.replace("_", " ").title() for m in metrics_to_plot],
                vertical_spacing=0.05
            )
            
            # Add traces for each metric
            for i, metric in enumerate(metrics_to_plot):
                # Extract values
                values = [m.get(metric, 0.0) for m in metrics_history]
                
                # Add main trace
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=values,
                        mode="lines+markers",
                        name=metric.replace("_", " ").title(),
                        line=dict(color=self.color_palette[i % len(self.color_palette)])
                    ),
                    row=i+1, col=1
                )
                
                # Add confidence interval
                if show_confidence_interval and len(values) > window_size:
                    # Calculate moving average and standard deviation
                    moving_avg = []
                    upper_bound = []
                    lower_bound = []
                    
                    for j in range(len(values)):
                        # Get window
                        start = max(0, j - window_size)
                        window = values[start:j+1]
                        
                        # Calculate statistics
                        avg = np.mean(window)
                        std = np.std(window)
                        
                        moving_avg.append(avg)
                        upper_bound.append(avg + std)
                        lower_bound.append(avg - std)
                    
                    # Add moving average
                    fig.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=moving_avg,
                            mode="lines",
                            line=dict(
                                color=self.color_palette[i % len(self.color_palette)],
                                dash="dash"
                            ),
                            name=f"{metric} (Moving Avg)",
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add confidence interval
                    fig.add_trace(
                        go.Scatter(
                            x=iterations + iterations[::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill="toself",
                            fillcolor=self.color_palette[i % len(self.color_palette)],
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            opacity=0.2
                        ),
                        row=i+1, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=300 * len(metrics_to_plot),
                width=1000
            )
            
            # Update x-axis titles
            fig.update_xaxes(title_text="Iteration", row=len(metrics_to_plot), col=1)
            
            # Save figure
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(png_path)
                
                return output_path
            else:
                # Use timestamp as filename
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(self.output_dir, f"metrics_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"metrics_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize metrics over time: {e}")
            return ""
    
    def visualize_domain_coverage(
        self,
        metrics_history: List[Dict[str, Any]],
        title: str = "Domain Coverage",
        filename: Optional[str] = None
    ) -> str:
        """Visualize domain coverage over time.
        
        Args:
            metrics_history: List of metrics at different iterations
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not metrics_history:
                log.warning("No metrics history provided")
                return ""
            
            # Extract iterations and domain coverage
            iterations = []
            domain_coverage_data = []
            
            for i, metrics in enumerate(metrics_history):
                if "domain_coverage" in metrics:
                    iterations.append(metrics.get("iteration", i))
                    domain_coverage_data.append(metrics["domain_coverage"])
            
            if not domain_coverage_data:
                log.warning("No domain coverage data found in metrics history")
                return ""
            
            # Get all domains
            all_domains = set()
            for coverage in domain_coverage_data:
                all_domains.update(coverage.keys())
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each domain
            for i, domain in enumerate(sorted(all_domains)):
                # Extract values
                values = [
                    coverage.get(domain, 0.0) 
                    for coverage in domain_coverage_data
                ]
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=values,
                        mode="lines+markers",
                        name=domain,
                        line=dict(color=self.color_palette[i % len(self.color_palette)])
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                xaxis_title="Iteration",
                yaxis_title="Coverage",
                yaxis=dict(range=[0, 1]),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=1000,
                height=600
            )
            
            # Save figure
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(png_path)
                
                return output_path
            else:
                # Use timestamp as filename
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(self.output_dir, f"domain_coverage_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"domain_coverage_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize domain coverage: {e}")
            return ""
    
    def visualize_community_coherence(
        self,
        metrics_history: List[Dict[str, Any]],
        title: str = "Community Coherence",
        filename: Optional[str] = None
    ) -> str:
        """Visualize community coherence over time.
        
        Args:
            metrics_history: List of metrics at different iterations
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not metrics_history:
                log.warning("No metrics history provided")
                return ""
            
            # Extract iterations and coherence metrics
            iterations = []
            avg_coherence = []
            min_coherence = []
            max_coherence = []
            
            for i, metrics in enumerate(metrics_history):
                if "avg_community_coherence" in metrics:
                    iterations.append(metrics.get("iteration", i))
                    avg_coherence.append(metrics["avg_community_coherence"])
                    min_coherence.append(metrics.get("min_community_coherence", 0.0))
                    max_coherence.append(metrics.get("max_community_coherence", 0.0))
            
            if not avg_coherence:
                log.warning("No community coherence data found in metrics history")
                return ""
            
            # Create figure
            fig = go.Figure()
            
            # Add average coherence
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=avg_coherence,
                    mode="lines+markers",
                    name="Average Coherence",
                    line=dict(color=self.color_palette[0])
                )
            )
            
            # Add min and max coherence as a range
            fig.add_trace(
                go.Scatter(
                    x=iterations + iterations[::-1],
                    y=max_coherence + min_coherence[::-1],
                    fill="toself",
                    fillcolor=self.color_palette[0],
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Coherence Range",
                    opacity=0.2
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                xaxis_title="Iteration",
                yaxis_title="Coherence",
                yaxis=dict(range=[0, 1]),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=1000,
                height=600
            )
            
            # Save figure
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(png_path)
                
                return output_path
            else:
                # Use timestamp as filename
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(self.output_dir, f"coherence_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"coherence_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize community coherence: {e}")
            return ""
    
    def visualize_comparison(
        self,
        comparison_data: Dict[str, Any],
        title: str = "Metrics Comparison",
        filename: Optional[str] = None
    ) -> str:
        """Visualize comparison between two iterations.
        
        Args:
            comparison_data: Comparison data from GraphEvaluator.compare_iterations
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if "error" in comparison_data:
                log.warning(f"Error in comparison data: {comparison_data['error']}")
                return ""
            
            if "metrics" not in comparison_data:
                log.warning("No metrics found in comparison data")
                return ""
            
            # Get iterations
            iterations = comparison_data.get("iterations", [0, 0])
            
            # Get metrics
            metrics = comparison_data["metrics"]
            
            # Create figure
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=("Metric Values", "Percent Change"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Prepare data
            metric_names = list(metrics.keys())
            values1 = [metrics[m]["values"][0] for m in metric_names]
            values2 = [metrics[m]["values"][1] for m in metric_names]
            pct_changes = [metrics[m]["percent_change"] for m in metric_names]
            
            # Format metric names for display
            display_names = [m.replace("_", " ").title() for m in metric_names]
            
            # Add value comparison
            fig.add_trace(
                go.Bar(
                    x=display_names,
                    y=values1,
                    name=f"Iteration {iterations[0]}",
                    marker_color=self.color_palette[0]
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=display_names,
                    y=values2,
                    name=f"Iteration {iterations[1]}",
                    marker_color=self.color_palette[1]
                ),
                row=1, col=1
            )
            
            # Add percent change
            fig.add_trace(
                go.Bar(
                    x=display_names,
                    y=pct_changes,
                    name="Percent Change",
                    marker_color=[
                        "green" if pct >= 0 else "red"
                        for pct in pct_changes
                    ]
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=1200,
                height=600
            )
            
            # Update y-axis titles
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Percent Change", row=1, col=2)
            
            # Save figure
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(png_path)
                
                return output_path
            else:
                # Use timestamp as filename
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(self.output_dir, f"comparison_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"comparison_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize comparison: {e}")
            return ""
    
    def visualize_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        title: str = "Detected Anomalies",
        filename: Optional[str] = None
    ) -> str:
        """Visualize detected anomalies.
        
        Args:
            anomalies: List of anomalies from GraphEvaluator.detect_anomalies
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not anomalies:
                log.warning("No anomalies provided")
                return ""
            
            # Create figure
            fig = go.Figure()
            
            # Prepare data
            metric_names = [a["metric"].replace("_", " ").title() for a in anomalies]
            values = [a["value"] for a in anomalies]
            means = [a["mean"] for a in anomalies]
            z_scores = [a["z_score"] for a in anomalies]
            
            # Add values and means
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=values,
                    name="Anomalous Value",
                    marker_color="red"
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=means,
                    name="Expected Value (Mean)",
                    marker_color="blue"
                )
            )
            
            # Add z-scores as text
            for i, z_score in enumerate(z_scores):
                fig.add_annotation(
                    x=metric_names[i],
                    y=max(values[i], means[i]) + 0.1,
                    text=f"z={z_score:.2f}",
                    showarrow=False,
                    font=dict(color="black", size=12)
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                xaxis_title="Metric",
                yaxis_title="Value",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=1000,
                height=600
            )
            
            # Save figure
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(png_path)
                
                return output_path
            else:
                # Use timestamp as filename
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(self.output_dir, f"anomalies_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"anomalies_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize anomalies: {e}")
            return ""
    
    def create_dashboard(
        self,
        metrics_history: List[Dict[str, Any]],
        title: str = "Knowledge Graph Dashboard",
        filename: Optional[str] = None
    ) -> str:
        """Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            metrics_history: List of metrics at different iterations
            title: Dashboard title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved dashboard
        """
        try:
            if not metrics_history:
                log.warning("No metrics history provided")
                return ""
            
            # Create output directory for dashboard
            dashboard_dir = os.path.join(self.output_dir, "dashboard")
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Generate individual visualizations
            metrics_viz_path = self.visualize_metrics_over_time(
                metrics_history,
                metrics_to_plot=["modularity", "avg_path_length", "diameter"],
                title="Core Metrics Over Time",
                filename=os.path.join(dashboard_dir, "core_metrics")
            )
            
            domain_viz_path = self.visualize_domain_coverage(
                metrics_history,
                title="Domain Coverage Over Time",
                filename=os.path.join(dashboard_dir, "domain_coverage")
            )
            
            coherence_viz_path = self.visualize_community_coherence(
                metrics_history,
                title="Community Coherence Over Time",
                filename=os.path.join(dashboard_dir, "community_coherence")
            )
            
            # Create HTML dashboard
            dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-title {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .viz-container {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .metrics-summary {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            width: 22%;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-summary">
"""
            
            # Add summary metrics from the latest iteration
            latest_metrics = metrics_history[-1]
            summary_metrics = [
                {"label": "Nodes", "value": latest_metrics.get("node_count", 0)},
                {"label": "Edges", "value": latest_metrics.get("edge_count", 0)},
                {"label": "Modularity", "value": f"{latest_metrics.get('modularity', 0.0):.2f}"},
                {"label": "Avg Path Length", "value": f"{latest_metrics.get('avg_path_length', 0.0):.2f}"}
            ]
            
            for metric in summary_metrics:
                dashboard_html += f"""
            <div class="metric-card">
                <div class="metric-label">{metric['label']}</div>
                <div class="metric-value">{metric['value']}</div>
            </div>"""
            
            dashboard_html += """
        </div>
        
        <div class="section">
            <h2 class="section-title">Core Metrics</h2>
            <iframe class="viz-container" src="core_metrics.html"></iframe>
        </div>
        
        <div class="section">
            <h2 class="section-title">Domain Coverage</h2>
            <iframe class="viz-container" src="domain_coverage.html"></iframe>
        </div>
        
        <div class="section">
            <h2 class="section-title">Community Coherence</h2>
            <iframe class="viz-container" src="community_coherence.html"></iframe>
        </div>
        
        <div class="footer">
            <p>Generated by Knowledge Graph Reasoner Evaluation Framework</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Save dashboard
            if filename:
                output_path = os.path.join(dashboard_dir, f"{filename}.html")
            else:
                timestamp = int(datetime.now().timestamp())
                output_path = os.path.join(dashboard_dir, f"dashboard_{timestamp}.html")
            
            with open(output_path, "w") as f:
                f.write(dashboard_html)
            
            return output_path
        except Exception as e:
            log.error(f"Failed to create dashboard: {e}")
            return ""
