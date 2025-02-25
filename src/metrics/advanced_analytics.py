"""Advanced analytics for knowledge graph reasoning."""
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import networkx as nx
import logging
import powerlaw
from collections import defaultdict, deque
from datetime import datetime

log = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced analytics for knowledge graph reasoning.
    
    Implements network metrics monitoring, community structure analysis,
    temporal evolution tracking, and scale-free property validation.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """Initialize advanced analytics.
        
        Args:
            graph: Optional NetworkX graph to analyze
        """
        self.graph = graph or nx.Graph()
        self.history: List[Dict[str, Any]] = []
        self.temporal_snapshots: List[Dict[str, Any]] = []
        self.target_metrics = {
            "assortativity": -0.05,  # Target degree assortativity
            "transitivity": 0.10,    # Target global transitivity
            "modularity": 0.69,      # Target modularity
            "avg_path_length": 4.75, # Target average path length (4.5-5.0)
            "diameter": 17.0,        # Target diameter (16-18)
            "power_law_exponent": 3.0 # Target power law exponent
        }
        self.tolerance = {
            "assortativity": 0.03,   # Tolerance for assortativity
            "transitivity": 0.02,    # Tolerance for transitivity
            "modularity": 0.05,      # Tolerance for modularity
            "avg_path_length": 0.25, # Tolerance for path length
            "diameter": 1.0,         # Tolerance for diameter
            "power_law_exponent": 0.2 # Tolerance for power law exponent
        }
    
    def set_graph(self, graph: nx.Graph) -> None:
        """Set the graph to analyze.
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
    
    async def compute_degree_assortativity(self) -> float:
        """Compute degree assortativity coefficient.
        
        Returns:
            float: Degree assortativity coefficient (-1 to 1)
        """
        try:
            if len(self.graph) < 3 or self.graph.number_of_edges() < 2:
                return 0.0
            
            return nx.degree_assortativity_coefficient(self.graph)
        except Exception as e:
            log.error(f"Failed to compute degree assortativity: {e}")
            return 0.0
    
    async def compute_global_transitivity(self) -> float:
        """Compute global transitivity (clustering coefficient).
        
        Returns:
            float: Global transitivity (0-1)
        """
        try:
            if len(self.graph) < 3:
                return 0.0
            
            return nx.transitivity(self.graph)
        except Exception as e:
            log.error(f"Failed to compute global transitivity: {e}")
            return 0.0
    
    async def compute_power_law_exponent(self) -> Tuple[float, float]:
        """Compute power law exponent for degree distribution.
        
        Returns:
            Tuple[float, float]: Power law exponent and goodness of fit
        """
        try:
            if len(self.graph) < 10:
                return 0.0, 0.0
            
            # Get degree sequence
            degrees = [d for _, d in self.graph.degree()]
            
            # Filter out zeros
            degrees = [d for d in degrees if d > 0]
            
            if not degrees:
                return 0.0, 0.0
            
            # Fit power law
            fit = powerlaw.Fit(degrees, discrete=True)
            
            # Get goodness of fit (R value)
            # Compare power law with exponential distribution
            r, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            
            return fit.alpha, r
        except Exception as e:
            log.error(f"Failed to compute power law exponent: {e}")
            return 0.0, 0.0
    
    async def compute_community_metrics(self) -> Dict[str, Any]:
        """Compute community structure metrics.
        
        Returns:
            Dict[str, Any]: Community metrics
        """
        try:
            if len(self.graph) < 3:
                return {
                    "modularity": 0.0,
                    "num_communities": 0,
                    "community_sizes": [],
                    "community_density": []
                }
            
            # Find communities using Louvain method
            communities = nx.community.louvain_communities(self.graph)
            
            # Compute modularity
            modularity = nx.community.modularity(self.graph, communities)
            
            # Compute community sizes
            community_sizes = [len(c) for c in communities]
            
            # Compute community density
            community_density = []
            for community in communities:
                subgraph = self.graph.subgraph(community)
                n = len(subgraph)
                if n <= 1:
                    density = 0.0
                else:
                    max_edges = n * (n - 1) / 2
                    density = subgraph.number_of_edges() / max_edges
                community_density.append(density)
            
            return {
                "modularity": modularity,
                "num_communities": len(communities),
                "community_sizes": community_sizes,
                "community_density": community_density
            }
        except Exception as e:
            log.error(f"Failed to compute community metrics: {e}")
            return {
                "modularity": 0.0,
                "num_communities": 0,
                "community_sizes": [],
                "community_density": []
            }
    
    async def compute_path_metrics(self) -> Dict[str, Any]:
        """Compute path-related metrics.
        
        Returns:
            Dict[str, Any]: Path metrics
        """
        try:
            if len(self.graph) < 2:
                return {
                    "avg_path_length": 0.0,
                    "diameter": 0.0,
                    "path_length_distribution": {}
                }
            
            # Only consider largest connected component
            components = list(nx.connected_components(self.graph))
            if not components:
                return {
                    "avg_path_length": 0.0,
                    "diameter": 0.0,
                    "path_length_distribution": {}
                }
            
            largest_cc = max(components, key=len)
            subgraph = self.graph.subgraph(largest_cc)
            
            # Compute average path length
            avg_path_length = nx.average_shortest_path_length(subgraph)
            
            # Compute diameter
            diameter = nx.diameter(subgraph)
            
            # Compute path length distribution
            distribution: Dict[int, int] = defaultdict(int)
            
            # Sample nodes if graph is large
            nodes_to_sample = min(len(subgraph), 100)
            sampled_nodes = np.random.choice(
                list(subgraph.nodes()),
                size=nodes_to_sample,
                replace=False
            )
            
            for source in sampled_nodes:
                path_lengths = nx.single_source_shortest_path_length(
                    subgraph,
                    source
                )
                for length in path_lengths.values():
                    distribution[length] += 1
            
            return {
                "avg_path_length": avg_path_length,
                "diameter": diameter,
                "path_length_distribution": dict(distribution)
            }
        except Exception as e:
            log.error(f"Failed to compute path metrics: {e}")
            return {
                "avg_path_length": 0.0,
                "diameter": 0.0,
                "path_length_distribution": {}
            }
    
    async def compute_hub_metrics(self) -> Dict[str, Any]:
        """Compute hub-related metrics.
        
        Returns:
            Dict[str, Any]: Hub metrics
        """
        try:
            if len(self.graph) < 3:
                return {
                    "hub_nodes": [],
                    "hub_centrality": {},
                    "hub_ratio": 0.0
                }
            
            # Compute centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph)
            except Exception:
                # Fall back to degree centrality if eigenvector fails
                eigenvector_centrality = degree_centrality
            
            # Combine centrality measures
            combined_centrality = {}
            for node in self.graph.nodes():
                combined_centrality[node] = (
                    0.4 * degree_centrality.get(node, 0) +
                    0.4 * eigenvector_centrality.get(node, 0) +
                    0.2 * betweenness_centrality.get(node, 0)
                )
            
            # Identify hub nodes (top 10%)
            threshold = np.percentile(
                list(combined_centrality.values()),
                90
            )
            
            hub_nodes = [
                node for node, score in combined_centrality.items()
                if score >= threshold
            ]
            
            # Calculate hub ratio
            hub_ratio = len(hub_nodes) / len(self.graph) if len(self.graph) > 0 else 0.0
            
            return {
                "hub_nodes": hub_nodes,
                "hub_centrality": combined_centrality,
                "hub_ratio": hub_ratio
            }
        except Exception as e:
            log.error(f"Failed to compute hub metrics: {e}")
            return {
                "hub_nodes": [],
                "hub_centrality": {},
                "hub_ratio": 0.0
            }
    
    async def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute all advanced analytics metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        assortativity = await self.compute_degree_assortativity()
        transitivity = await self.compute_global_transitivity()
        power_law_exponent, power_law_fit = await self.compute_power_law_exponent()
        community_metrics = await self.compute_community_metrics()
        path_metrics = await self.compute_path_metrics()
        hub_metrics = await self.compute_hub_metrics()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "graph_size": len(self.graph),
            "edge_count": self.graph.number_of_edges(),
            "assortativity": assortativity,
            "transitivity": transitivity,
            "power_law_exponent": power_law_exponent,
            "power_law_fit": power_law_fit,
            "modularity": community_metrics["modularity"],
            "num_communities": community_metrics["num_communities"],
            "community_sizes": community_metrics["community_sizes"],
            "community_density": community_metrics["community_density"],
            "avg_path_length": path_metrics["avg_path_length"],
            "diameter": path_metrics["diameter"],
            "path_length_distribution": path_metrics["path_length_distribution"],
            "hub_nodes": hub_metrics["hub_nodes"],
            "hub_ratio": hub_metrics["hub_ratio"]
        }
        
        return metrics
    
    async def track_metrics(self, snapshot_graph: bool = False) -> Dict[str, Any]:
        """Compute and track metrics over time.
        
        Args:
            snapshot_graph: Whether to save a snapshot of the graph
            
        Returns:
            Dict[str, Any]: Current metrics
        """
        metrics = await self.compute_all_metrics()
        self.history.append(metrics)
        
        if snapshot_graph and len(self.graph) > 0:
            # Create a deep copy of the graph for the snapshot
            snapshot = {
                "timestamp": metrics["timestamp"],
                "graph": nx.Graph(self.graph),
                "metrics": metrics
            }
            self.temporal_snapshots.append(snapshot)
        
        return metrics
    
    async def check_convergence(self, window_size: int = 5) -> Dict[str, bool]:
        """Check if metrics have converged to target values.
        
        Args:
            window_size: Number of historical states to consider
            
        Returns:
            Dict[str, bool]: Convergence status for each metric
        """
        if len(self.history) < window_size:
            return {metric: False for metric in self.target_metrics}
        
        # Get recent metrics
        recent = self.history[-window_size:]
        
        # Check convergence for each metric
        convergence = {}
        for metric, target in self.target_metrics.items():
            if metric not in recent[0]:
                convergence[metric] = False
                continue
            
            values = [m.get(metric, 0.0) for m in recent]
            
            # Check if values are within tolerance of target
            tolerance = self.tolerance.get(metric, 0.05)
            converged = all(
                abs(value - target) <= tolerance
                for value in values
            )
            
            # Also check stability (low variance)
            if len(values) > 1:
                variance = np.var(values)
                stable = variance < (tolerance / 2) ** 2
                converged = converged and stable
            
            convergence[metric] = converged
        
        return convergence
    
    async def analyze_temporal_evolution(
        self,
        metric: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze temporal evolution of a specific metric.
        
        Args:
            metric: Metric to analyze
            window_size: Window size for trend analysis
            
        Returns:
            Dict[str, Any]: Temporal analysis results
        """
        if len(self.history) < 2:
            return {
                "trend": "unknown",
                "stability": 0.0,
                "rate_of_change": 0.0
            }
        
        # Get values for the metric
        values = [
            h.get(metric, 0.0)
            for h in self.history
            if metric in h
        ]
        
        if not values:
            return {
                "trend": "unknown",
                "stability": 0.0,
                "rate_of_change": 0.0
            }
        
        # Compute trend
        window = min(window_size, len(values))
        recent_values = values[-window:]
        
        if len(recent_values) < 2:
            trend = "stable"
        else:
            # Linear regression to find trend
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            
            # Fit line: y = mx + b
            m, b = np.polyfit(x, y, 1)
            
            # Determine trend based on slope
            if abs(m) < 0.001:
                trend = "stable"
            elif m > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
        
        # Compute stability (inverse of variance)
        stability = 1.0 / (np.var(recent_values) + 0.0001)
        
        # Compute rate of change
        if len(values) < 2:
            rate_of_change = 0.0
        else:
            rate_of_change = (values[-1] - values[0]) / len(values)
        
        return {
            "trend": trend,
            "stability": min(stability, 100.0),  # Cap at 100
            "rate_of_change": rate_of_change
        }
    
    async def validate_scale_free_property(self) -> Dict[str, Any]:
        """Validate if the graph exhibits scale-free properties.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            if len(self.graph) < 20:
                return {
                    "is_scale_free": False,
                    "power_law_exponent": 0.0,
                    "power_law_fit": 0.0,
                    "degree_distribution": {}
                }
            
            # Get degree sequence
            degrees = [d for _, d in self.graph.degree()]
            
            # Compute degree distribution
            degree_counts = defaultdict(int)
            for d in degrees:
                degree_counts[d] += 1
            
            # Normalize to get probability distribution
            total_nodes = len(self.graph)
            degree_distribution = {
                d: count / total_nodes
                for d, count in degree_counts.items()
            }
            
            # Compute power law exponent
            exponent, fit = await self.compute_power_law_exponent()
            
            # Check if it follows power law
            # R > 0 means power law is favored over exponential
            is_scale_free = exponent > 2.0 and fit > 0
            
            return {
                "is_scale_free": is_scale_free,
                "power_law_exponent": exponent,
                "power_law_fit": fit,
                "degree_distribution": degree_distribution
            }
        except Exception as e:
            log.error(f"Failed to validate scale-free property: {e}")
            return {
                "is_scale_free": False,
                "power_law_exponent": 0.0,
                "power_law_fit": 0.0,
                "degree_distribution": {}
            }
    
    async def get_metric_history(self, metric: str) -> List[float]:
        """Get historical values for a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List[float]: Historical values
        """
        return [
            h.get(metric, 0.0)
            for h in self.history
            if metric in h
        ]
    
    async def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all metrics.
        
        Returns:
            Dict[str, Any]: Summary report
        """
        if not self.history:
            return {
                "status": "No data available",
                "metrics": {},
                "convergence": {},
                "recommendations": []
            }
        
        # Get latest metrics
        current = self.history[-1]
        
        # Check convergence
        convergence = await self.check_convergence()
        
        # Generate recommendations
        recommendations = []
        
        # Check assortativity
        if "assortativity" in current:
            target = self.target_metrics["assortativity"]
            current_value = current["assortativity"]
            if abs(current_value - target) > self.tolerance["assortativity"]:
                if current_value < target:
                    recommendations.append(
                        "Increase connections between similar-degree nodes"
                    )
                else:
                    recommendations.append(
                        "Increase connections between hub nodes and peripheral nodes"
                    )
        
        # Check transitivity
        if "transitivity" in current:
            target = self.target_metrics["transitivity"]
            current_value = current["transitivity"]
            if abs(current_value - target) > self.tolerance["transitivity"]:
                if current_value < target:
                    recommendations.append(
                        "Increase local clustering by adding connections within communities"
                    )
                else:
                    recommendations.append(
                        "Reduce excessive clustering by adding cross-community connections"
                    )
        
        # Check modularity
        if "modularity" in current:
            target = self.target_metrics["modularity"]
            current_value = current["modularity"]
            if abs(current_value - target) > self.tolerance["modularity"]:
                if current_value < target:
                    recommendations.append(
                        "Strengthen community structure by adding intra-community connections"
                    )
                else:
                    recommendations.append(
                        "Add more bridge nodes to connect communities"
                    )
        
        # Check path length
        if "avg_path_length" in current:
            target = self.target_metrics["avg_path_length"]
            current_value = current["avg_path_length"]
            if abs(current_value - target) > self.tolerance["avg_path_length"]:
                if current_value < target:
                    recommendations.append(
                        "Reduce shortcuts between distant nodes"
                    )
                else:
                    recommendations.append(
                        "Add strategic connections to reduce average path length"
                    )
        
        # Check power law exponent
        if "power_law_exponent" in current:
            target = self.target_metrics["power_law_exponent"]
            current_value = current["power_law_exponent"]
            if abs(current_value - target) > self.tolerance["power_law_exponent"]:
                if current_value < target:
                    recommendations.append(
                        "Strengthen hub nodes by adding more connections to them"
                    )
                else:
                    recommendations.append(
                        "Add more connections between non-hub nodes"
                    )
        
        # Overall status
        all_converged = all(convergence.values())
        if all_converged:
            status = "Optimal: All metrics have converged to target values"
        elif any(convergence.values()):
            status = "Improving: Some metrics have converged to target values"
        else:
            status = "Developing: No metrics have converged to target values yet"
        
        return {
            "status": status,
            "metrics": current,
            "convergence": convergence,
            "recommendations": recommendations
        }
