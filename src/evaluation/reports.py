"""Report generation module for knowledge graph reasoner."""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
import os
import logging
import json
import time
from pathlib import Path
import networkx as nx
import numpy as np
from datetime import datetime
import jinja2
import markdown
import base64

from ..models.node import Node
from ..models.edge import Edge
from ..graph.manager import GraphManager
from .metrics import GraphEvaluator
from ..visualization.graph_viz import GraphVisualizer
from ..visualization.metrics_viz import MetricsVisualizer

log = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports for knowledge graph evaluation."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        evaluator: GraphEvaluator,
        output_dir: Optional[str] = None,
        template_dir: Optional[str] = None
    ):
        """Initialize report generator.
        
        Args:
            graph_manager: Graph manager instance
            evaluator: Graph evaluator instance
            output_dir: Directory to save reports
            template_dir: Directory containing report templates
        """
        self.graph_manager = graph_manager
        self.evaluator = evaluator
        self.output_dir = output_dir or "reports"
        
        # Default template directory is in the same package
        if template_dir:
            self.template_dir = template_dir
        else:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.template_dir = os.path.join(current_dir, "templates")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html'])
        )
    
    async def generate_markdown_report(
        self,
        title: str = "Knowledge Graph Report",
        include_visualizations: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """Generate Markdown report.
        
        Args:
            title: Report title
            include_visualizations: Whether to include visualizations
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved report
        """
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Get evaluation metrics
            metrics = await self.evaluator.evaluate(state.get("iteration", 0))
            
            # Get evaluation history
            history = await self.evaluator.get_history()
            
            # Create visualizations if requested
            visualization_paths = {}
            if include_visualizations:
                # Create visualization directory
                viz_dir = os.path.join(self.output_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                # Create visualizers
                graph_viz = GraphVisualizer(self.graph_manager, output_dir=viz_dir)
                metrics_viz = MetricsVisualizer(output_dir=viz_dir)
                
                # Generate visualizations
                graph_path = await graph_viz.visualize_graph(
                    title="Knowledge Graph Structure",
                    filename="graph_structure"
                )
                visualization_paths["graph"] = os.path.relpath(graph_path, self.output_dir)
                
                community_path = await graph_viz.visualize_communities(
                    title="Community Structure",
                    filename="community_structure"
                )
                visualization_paths["communities"] = os.path.relpath(community_path, self.output_dir)
                
                if history:
                    metrics_path = metrics_viz.visualize_metrics_over_time(
                        history,
                        title="Metrics Over Time",
                        filename="metrics_over_time"
                    )
                    visualization_paths["metrics"] = os.path.relpath(metrics_path, self.output_dir)
            
            # Prepare template data
            template_data = {
                "report_title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "node_count": state.get("node_count", 0),
                "edge_count": state.get("edge_count", 0),
                "modularity": state.get("modularity", 0.0),
                "avg_path_length": state.get("avg_path_length", 0.0),
                "diameter": state.get("diameter", 0.0),
                "bridge_node_count": len(state.get("bridge_nodes", [])),
                "visualization_paths": visualization_paths,
                "metrics": metrics,
                "history": history
            }
            
            # Generate executive summary
            template_data["executive_summary"] = self._generate_executive_summary(template_data)
            
            # Generate community structure section
            template_data["community_structure"] = self._generate_community_structure(state)
            
            # Generate domain coverage section
            template_data["domain_coverage"] = self._generate_domain_coverage(metrics)
            
            # Generate evaluation metrics section
            template_data["evaluation_metrics"] = self._generate_evaluation_metrics(metrics)
            
            # Generate anomalies and insights section
            template_data["anomalies_and_insights"] = self._generate_anomalies_and_insights(
                metrics, history
            )
            
            # Generate recommendations section
            template_data["recommendations"] = self._generate_recommendations(
                metrics, history, state
            )
            
            # Render template
            template = self.jinja_env.get_template("report_template.md")
            report_content = template.render(**template_data)
            
            # Save report
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.md")
            else:
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"report_{timestamp}.md")
            
            with open(output_path, "w") as f:
                f.write(report_content)
            
            return output_path
        except Exception as e:
            log.error(f"Failed to generate Markdown report: {e}")
            return ""
    
    async def generate_html_report(
        self,
        title: str = "Knowledge Graph Report",
        include_visualizations: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """Generate HTML report.
        
        Args:
            title: Report title
            include_visualizations: Whether to include visualizations
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved report
        """
        try:
            # First generate Markdown report
            md_path = await self.generate_markdown_report(
                title=title,
                include_visualizations=include_visualizations,
                filename=f"{filename}_md" if filename else None
            )
            
            if not md_path:
                return ""
            
            # Read Markdown content
            with open(md_path, "r") as f:
                md_content = f.read()
            
            # Convert Markdown to HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['tables', 'fenced_code']
            )
            
            # Add HTML styling
            styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        code {{
            background-color: #f8f8f8;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        pre {{
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .visualization {{
            max-width: 100%;
            margin: 20px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.8em;
            color: #777;
        }}
    </style>
</head>
<body>
    {html_content}
    <div class="footer">
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by Knowledge Graph Reasoner</p>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"report_{timestamp}.html")
            
            with open(output_path, "w") as f:
                f.write(styled_html)
            
            return output_path
        except Exception as e:
            log.error(f"Failed to generate HTML report: {e}")
            return ""
    
    async def generate_comparison_report(
        self,
        iter1: int,
        iter2: int,
        title: str = "Knowledge Graph Comparison Report",
        filename: Optional[str] = None
    ) -> str:
        """Generate comparison report between two iterations.
        
        Args:
            iter1: First iteration
            iter2: Second iteration
            title: Report title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved report
        """
        try:
            # Get comparison data
            comparison = await self.evaluator.compare_iterations(iter1, iter2)
            
            if "error" in comparison:
                log.error(f"Failed to compare iterations: {comparison['error']}")
                return ""
            
            # Create visualization directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create metrics visualizer
            metrics_viz = MetricsVisualizer(output_dir=viz_dir)
            
            # Generate comparison visualization
            viz_path = metrics_viz.visualize_comparison(
                comparison,
                title=f"Comparison: Iteration {iter1} vs {iter2}",
                filename=f"comparison_{iter1}_{iter2}"
            )
            
            # Prepare template data
            template_data = {
                "report_title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comparison": comparison,
                "iter1": iter1,
                "iter2": iter2,
                "visualization_path": os.path.relpath(viz_path, self.output_dir) if viz_path else None
            }
            
            # Generate comparison summary
            template_data["comparison_summary"] = self._generate_comparison_summary(comparison)
            
            # Render template
            template = self.jinja_env.get_template("comparison_template.md")
            report_content = template.render(**template_data)
            
            # Save report
            if filename:
                output_path = os.path.join(self.output_dir, f"{filename}.md")
            else:
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"comparison_{iter1}_{iter2}_{timestamp}.md")
            
            with open(output_path, "w") as f:
                f.write(report_content)
            
            return output_path
        except Exception as e:
            log.error(f"Failed to generate comparison report: {e}")
            return ""
    
    async def generate_dashboard(
        self,
        title: str = "Knowledge Graph Dashboard",
        filename: Optional[str] = None
    ) -> str:
        """Generate interactive dashboard.
        
        Args:
            title: Dashboard title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved dashboard
        """
        try:
            # Get evaluation history
            history = await self.evaluator.get_history()
            
            if not history:
                log.warning("No evaluation history available for dashboard")
                return ""
            
            # Create visualization directory
            viz_dir = os.path.join(self.output_dir, "dashboard")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create visualizers
            metrics_viz = MetricsVisualizer(output_dir=viz_dir)
            
            # Generate dashboard
            dashboard_path = metrics_viz.create_dashboard(
                history,
                title=title,
                filename=filename or f"dashboard_{int(time.time())}"
            )
            
            return dashboard_path
        except Exception as e:
            log.error(f"Failed to generate dashboard: {e}")
            return ""
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section.
        
        Args:
            data: Template data
            
        Returns:
            str: Executive summary
        """
        # Extract key metrics
        node_count = data.get("node_count", 0)
        edge_count = data.get("edge_count", 0)
        modularity = data.get("modularity", 0.0)
        
        # Get community count
        community_count = 0
        if "metrics" in data and "community_count" in data["metrics"]:
            community_count = data["metrics"]["community_count"]
        
        # Generate summary
        summary = f"""
This report provides an analysis of the knowledge graph containing {node_count} concepts (nodes) and {edge_count} relationships (edges). 
The graph has a modularity score of {modularity:.2f}, indicating {'a strong' if modularity > 0.7 else 'a moderate' if modularity > 0.4 else 'a weak'} community structure.

The knowledge graph contains {community_count} distinct communities, representing different domains or clusters of related concepts.
"""
        
        # Add domain coverage if available
        if "metrics" in data and "domain_coverage" in data["metrics"]:
            domain_coverage = data["metrics"]["domain_coverage"]
            if domain_coverage:
                summary += "\n\nDomain coverage analysis shows that the graph includes concepts from the following domains:\n"
                for domain, coverage in domain_coverage.items():
                    summary += f"- {domain}: {coverage:.1%} coverage\n"
        
        # Add bridge node information
        bridge_node_count = data.get("bridge_node_count", 0)
        summary += f"\n\nThe graph contains {bridge_node_count} bridge nodes that connect different communities, facilitating cross-domain knowledge integration."
        
        # Add insights from history if available
        if "history" in data and len(data["history"]) > 1:
            first = data["history"][0]
            last = data["history"][-1]
            
            # Calculate growth
            node_growth = last.get("node_count", 0) - first.get("node_count", 0)
            edge_growth = last.get("edge_count", 0) - first.get("edge_count", 0)
            
            summary += f"\n\nOver the course of {len(data['history'])} iterations, the graph has grown by {node_growth} nodes and {edge_growth} edges."
        
        return summary
    
    def _generate_community_structure(self, state: Dict[str, Any]) -> str:
        """Generate community structure section.
        
        Args:
            state: Graph state
            
        Returns:
            str: Community structure description
        """
        communities = state.get("communities", [])
        
        if not communities:
            return "No community structure detected in the graph."
        
        # Generate description
        description = f"The graph contains {len(communities)} communities:\n\n"
        
        for i, community in enumerate(communities):
            description += f"### Community {i+1}\n\n"
            description += f"- **Size**: {len(community)} nodes\n"
            
            # Add top nodes if available
            if "centrality" in state:
                centrality = state["centrality"]
                
                # Get nodes in this community with centrality
                community_nodes = []
                for node_id in community:
                    if node_id in centrality:
                        community_nodes.append((node_id, centrality[node_id]))
                
                # Sort by centrality
                community_nodes.sort(key=lambda x: x[1], reverse=True)
                
                # Add top 5 nodes
                if community_nodes:
                    description += "- **Key concepts**:\n"
                    for node_id, centrality in community_nodes[:5]:
                        description += f"  - {node_id} (centrality: {centrality:.3f})\n"
            
            description += "\n"
        
        return description
    
    def _generate_domain_coverage(self, metrics: Dict[str, Any]) -> str:
        """Generate domain coverage section.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            str: Domain coverage description
        """
        if "domain_coverage" not in metrics:
            return "No domain coverage information available."
        
        domain_coverage = metrics["domain_coverage"]
        
        if not domain_coverage:
            return "No domain coverage information available."
        
        # Generate description
        description = "The knowledge graph covers the following domains:\n\n"
        
        for domain, coverage in domain_coverage.items():
            description += f"- **{domain}**: {coverage:.1%} coverage\n"
        
        return description
    
    def _generate_evaluation_metrics(self, metrics: Dict[str, Any]) -> str:
        """Generate evaluation metrics section.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            str: Evaluation metrics description
        """
        # Generate description
        description = "### Core Metrics\n\n"
        
        # Add core metrics
        core_metrics = [
            ("modularity", "Modularity"),
            ("avg_path_length", "Average Path Length"),
            ("diameter", "Diameter"),
            ("density", "Density"),
            ("clustering_coefficient", "Clustering Coefficient")
        ]
        
        for key, label in core_metrics:
            if key in metrics:
                description += f"- **{label}**: {metrics[key]:.3f}\n"
        
        # Add community metrics
        if "avg_community_coherence" in metrics:
            description += "\n### Community Metrics\n\n"
            description += f"- **Average Community Coherence**: {metrics['avg_community_coherence']:.3f}\n"
            
            if "min_community_coherence" in metrics:
                description += f"- **Minimum Community Coherence**: {metrics['min_community_coherence']:.3f}\n"
            
            if "max_community_coherence" in metrics:
                description += f"- **Maximum Community Coherence**: {metrics['max_community_coherence']:.3f}\n"
        
        # Add interdisciplinary metrics
        interdisciplinary_metrics = [
            ("cross_community_edges", "Cross-Community Edges"),
            ("interdisciplinary_ratio", "Interdisciplinary Ratio"),
            ("bridge_node_centrality", "Bridge Node Centrality")
        ]
        
        has_interdisciplinary = any(key in metrics for key, _ in interdisciplinary_metrics)
        
        if has_interdisciplinary:
            description += "\n### Interdisciplinary Metrics\n\n"
            
            for key, label in interdisciplinary_metrics:
                if key in metrics:
                    description += f"- **{label}**: {metrics[key]:.3f}\n"
        
        # Add novelty metrics
        if "avg_novelty" in metrics:
            description += "\n### Novelty Metrics\n\n"
            description += f"- **Average Novelty Score**: {metrics['avg_novelty']:.3f}\n"
        
        return description
    
    def _generate_anomalies_and_insights(
        self,
        metrics: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate anomalies and insights section.
        
        Args:
            metrics: Evaluation metrics
            history: Evaluation history
            
        Returns:
            str: Anomalies and insights description
        """
        if not history or len(history) < 2:
            return "Insufficient history to generate insights."
        
        # Generate description
        description = ""
        
        # Check for anomalies
        if "anomalies" in metrics and metrics["anomalies"]:
            anomalies = metrics["anomalies"]
            description += "### Detected Anomalies\n\n"
            
            for anomaly in anomalies:
                metric = anomaly.get("metric", "unknown")
                value = anomaly.get("value", 0)
                mean = anomaly.get("mean", 0)
                z_score = anomaly.get("z_score", 0)
                
                description += f"- **{metric}**: Current value ({value:.3f}) deviates significantly from the mean ({mean:.3f}), z-score: {z_score:.2f}\n"
            
            description += "\n"
        
        # Generate insights from history
        description += "### Growth Trends\n\n"
        
        # Calculate growth rates
        first = history[0]
        last = history[-1]
        
        metrics_to_track = [
            ("node_count", "Nodes"),
            ("edge_count", "Edges"),
            ("modularity", "Modularity"),
            ("avg_path_length", "Average Path Length")
        ]
        
        for key, label in metrics_to_track:
            if key in first and key in last:
                start_value = first.get(key, 0)
                end_value = last.get(key, 0)
                
                if start_value != 0:
                    change_pct = (end_value - start_value) / start_value * 100
                    description += f"- **{label}**: {change_pct:+.1f}% change over {len(history)} iterations\n"
        
        # Add community evolution insights
        if len(history) > 2:
            community_counts = [h.get("community_count", 0) for h in history if "community_count" in h]
            
            if community_counts:
                description += "\n### Community Evolution\n\n"
                
                # Check if communities are merging or splitting
                if community_counts[0] < community_counts[-1]:
                    description += f"- Communities are **splitting** over time (from {community_counts[0]} to {community_counts[-1]})\n"
                elif community_counts[0] > community_counts[-1]:
                    description += f"- Communities are **merging** over time (from {community_counts[0]} to {community_counts[-1]})\n"
                else:
                    description += f"- Community count is **stable** at {community_counts[-1]}\n"
                
                # Check coherence trends
                coherence_values = [h.get("avg_community_coherence", 0) for h in history if "avg_community_coherence" in h]
                
                if coherence_values and len(coherence_values) > 2:
                    if coherence_values[-1] > coherence_values[0]:
                        description += "- Community coherence is **increasing** over time\n"
                    else:
                        description += "- Community coherence is **decreasing** over time\n"
        
        return description
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        history: List[Dict[str, Any]],
        state: Dict[str, Any]
    ) -> str:
        """Generate recommendations section.
        
        Args:
            metrics: Evaluation metrics
            history: Evaluation history
            state: Graph state
            
        Returns:
            str: Recommendations description
        """
        # Generate description
        description = ""
        
        # Recommend based on modularity
        modularity = state.get("modularity", 0.0)
        
        if modularity < 0.3:
            description += "- **Improve community structure**: The graph has a weak community structure. Consider adding more domain-specific concepts to strengthen communities.\n"
        
        # Recommend based on path length
        avg_path_length = state.get("avg_path_length", 0.0)
        
        if avg_path_length > 4.0:
            description += "- **Add more connections**: The average path length is high. Consider adding more relationships between distant concepts to improve connectivity.\n"
        
        # Recommend based on bridge nodes
        bridge_nodes = state.get("bridge_nodes", [])
        
        if len(bridge_nodes) < 5:
            description += "- **Increase interdisciplinary connections**: The graph has few bridge nodes. Focus on adding concepts that connect different domains.\n"
        
        # Recommend based on domain coverage
        if "domain_coverage" in metrics:
            domain_coverage = metrics["domain_coverage"]
            
            low_coverage_domains = []
            for domain, coverage in domain_coverage.items():
                if coverage < 0.2:
                    low_coverage_domains.append(domain)
            
            if low_coverage_domains:
                domains_str = ", ".join(low_coverage_domains)
                description += f"- **Expand domain coverage**: The following domains have low coverage: {domains_str}. Consider adding more concepts in these areas.\n"
        
        # Recommend based on community coherence
        if "avg_community_coherence" in metrics and metrics["avg_community_coherence"] < 0.6:
            description += "- **Improve semantic coherence**: Community coherence is low. Consider refining concept definitions and relationships to improve semantic consistency.\n"
        
        # Add general recommendations
        description += "- **Regular evaluation**: Continue monitoring graph metrics to track the evolution of the knowledge structure.\n"
        description += "- **Targeted expansion**: Focus expansion on areas with high novelty potential to discover new insights.\n"
        
        return description
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate comparison summary section.
        
        Args:
            comparison: Comparison data
            
        Returns:
            str: Comparison summary
        """
        iterations = comparison.get("iterations", [0, 0])
        iter1, iter2 = iterations
        
        # Generate summary
        summary = f"""
This report compares the knowledge graph between iteration {iter1} and iteration {iter2}.
"""
        
        # Add metrics comparison
        if "metrics" in comparison:
            metrics = comparison["metrics"]
            
            # Add key metrics
            key_metrics = [
                ("node_count", "Nodes"),
                ("edge_count", "Edges"),
                ("modularity", "Modularity"),
                ("avg_path_length", "Average Path Length")
            ]
            
            summary += "\n### Key Metrics Comparison\n\n"
            summary += "| Metric | Iteration {} | Iteration {} | Change | % Change |\n".format(iter1, iter2)
            summary += "|--------|--------------|--------------|--------|----------|\n"
            
            for key, label in key_metrics:
                if key in metrics:
                    metric = metrics[key]
                    val1, val2 = metric["values"]
                    diff = metric["difference"]
                    pct = metric["percent_change"]
                    
                    summary += "| {} | {:.3f} | {:.3f} | {:+.3f} | {:+.1f}% |\n".format(
                        label, val1, val2, diff, pct
                    )
        
        # Add domain coverage comparison
        if "domain_coverage" in comparison:
            domain_coverage = comparison["domain_coverage"]
            
            summary += "\n### Domain Coverage Comparison\n\n"
            summary += "| Domain | Iteration {} | Iteration {} | Change | % Change |\n".format(iter1, iter2)
            summary += "|--------|--------------|--------------|--------|----------|\n"
            
            for domain, data in domain_coverage.items():
                val1, val2 = data["values"]
                diff = data["difference"]
                pct = data["percent_change"]
                
                summary += "| {} | {:.1%} | {:.1%} | {:+.1%} | {:+.1f}% |\n".format(
                    domain, val1, val2, diff, pct
                )
        
        # Add insights
        summary += "\n### Key Insights\n\n"
        
        # Add growth insights
        if "metrics" in comparison and "node_count" in comparison["metrics"]:
            node_metric = comparison["metrics"]["node_count"]
            node_growth = node_metric["difference"]
            
            if "edge_count" in comparison["metrics"]:
                edge_metric = comparison["metrics"]["edge_count"]
                edge_growth = edge_metric["difference"]
                
                summary += f"- The graph grew by {node_growth} nodes and {edge_growth} edges between iterations.\n"
        
        # Add modularity insights
        if "metrics" in comparison and "modularity" in comparison["metrics"]:
            mod_metric = comparison["metrics"]["modularity"]
            mod_diff = mod_metric["difference"]
            
            if mod_diff > 0:
                summary += f"- Community structure has **strengthened** (modularity increased by {mod_diff:.3f}).\n"
            elif mod_diff < 0:
                summary += f"- Community structure has **weakened** (modularity decreased by {abs(mod_diff):.3f}).\n"
            else:
                summary += "- Community structure has remained **stable**.\n"
        
        return summary
