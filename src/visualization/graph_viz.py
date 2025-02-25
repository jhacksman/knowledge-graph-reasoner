"""Graph visualization module for knowledge graph reasoner."""
from typing import List, Dict, Any, Optional, Tuple, Set
import os
import logging
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

from ..models.node import Node
from ..models.edge import Edge
from ..graph.manager import GraphManager

log = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualizes knowledge graphs and their properties."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        output_dir: Optional[str] = None,
        color_palette: Optional[List[str]] = None
    ):
        """Initialize visualizer.
        
        Args:
            graph_manager: Graph manager instance
            output_dir: Directory to save visualizations
            color_palette: Custom color palette for communities
        """
        self.graph_manager = graph_manager
        self.output_dir = output_dir or "visualizations"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default color palette
        self.color_palette = color_palette or [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
    
    async def visualize_graph(
        self,
        title: str = "Knowledge Graph",
        highlight_bridge_nodes: bool = True,
        show_communities: bool = True,
        node_size_by_centrality: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """Create interactive graph visualization.
        
        Args:
            title: Visualization title
            highlight_bridge_nodes: Whether to highlight bridge nodes
            show_communities: Whether to color nodes by community
            node_size_by_centrality: Whether to size nodes by centrality
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Create NetworkX graph
            G: nx.Graph = nx.Graph()
            
            # Get all nodes and edges
            nodes = []
            for node in await self.graph_manager.get_concept():
                nodes.append(node)
                G.add_node(node and node.id, content=node and node.content, metadata=node and node.metadata)
            
            edges = []
            for edge in await self.graph_manager.get_relationship():
                edges.append(edge)
                G.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    metadata=edge.metadata
                )
            
            # Get communities
            communities = state.get("communities", [])
            
            # Get bridge nodes
            bridge_nodes = set(state.get("bridge_nodes", []))
            
            # Get centrality
            centrality = state.get("centrality", {})
            if not centrality and node_size_by_centrality:
                # Compute centrality if not available
                centrality = nx.eigenvector_centrality_numpy(G, weight=None)
            
            # Create node-to-community mapping
            node_community = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    node_community[node_id] = i
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            # Use force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            for node_id in G.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                # Node text
                node_content = G.nodes[node_id].get("content", "")
                node_text.append(f"{node_id}: {node_content[:50]}...")
                
                # Node size
                if node_size_by_centrality:
                    size = 10 + 40 * centrality.get(node_id, 0.1)
                else:
                    size = 15
                node_size.append(size)
                
                # Node color
                if node_id in bridge_nodes and highlight_bridge_nodes:
                    # Bridge nodes are red
                    node_color.append("red")
                elif show_communities and node_id in node_community:
                    # Color by community
                    community_idx = node_community[node_id]
                    color_idx = community_idx % len(self.color_palette)
                    node_color.append(self.color_palette[color_idx])
                else:
                    # Default color
                    node_color.append("blue")
            
            # Create edge traces
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in edges:
                if edge.source in pos and edge.target in pos:
                    x0, y0 = pos[edge.source]
                    x1, y1 = pos[edge.target]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_text.append(f"{edge.type}")
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
                name="Relationships"
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                hovertext=node_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=1, color="#333")
                ),
                name="Concepts"
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1000,
                height=800
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
                import time
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"graph_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"graph_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize graph: {e}")
            return ""
    
    async def visualize_communities(
        self,
        title: str = "Community Structure",
        filename: Optional[str] = None
    ) -> str:
        """Visualize community structure.
        
        Args:
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Get communities
            communities = state.get("communities", [])
            if not communities:
                log.warning("No communities found")
                return ""
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Community Sizes", "Community Connections"),
                specs=[[{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # Community sizes
            community_sizes = [len(community) for community in communities]
            community_labels = [f"Community {i+1}" for i in range(len(communities))]
            
            fig.add_trace(
                go.Bar(
                    x=community_labels,
                    y=community_sizes,
                    marker_color=self.color_palette[:len(communities)],
                    name="Community Size"
                ),
                row=1, col=1
            )
            
            # Community connections
            connection_matrix = np.zeros((len(communities), len(communities)))
            
            # Get all edges
            edges = []
            for edge in await self.graph_manager.get_relationship():
                edges.append(edge)
            
            # Create node-to-community mapping
            node_community = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    node_community[node_id] = i
            
            # Count connections between communities
            for edge in edges:
                if edge.source in node_community and edge.target in node_community:
                    source_comm = node_community[edge.source]
                    target_comm = node_community[edge.target]
                    connection_matrix[source_comm][target_comm] += 1
                    connection_matrix[target_comm][source_comm] += 1
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=connection_matrix,
                    x=community_labels,
                    y=community_labels,
                    colorscale="Viridis",
                    name="Connections"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                width=1200,
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
                import time
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"communities_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"communities_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize communities: {e}")
            return ""
    
    async def visualize_bridge_nodes(
        self,
        title: str = "Bridge Node Analysis",
        filename: Optional[str] = None
    ) -> str:
        """Visualize bridge nodes.
        
        Args:
            title: Visualization title
            filename: Output filename (without extension)
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get graph state
            state = await self.graph_manager.get_graph_state()
            
            # Get bridge nodes
            bridge_nodes = state.get("bridge_nodes", [])
            if not bridge_nodes:
                log.warning("No bridge nodes found")
                return ""
            
            # Get centrality
            centrality = state.get("centrality", {})
            
            # Get bridge node details
            bridge_details = []
            for node_id in bridge_nodes:
                node = await self.graph_manager.get_concept(node_id)
                if node:
                    bridge_details.append({
                        "id": node_id,
                        "content": node and node.content,
                        "centrality": centrality.get(node_id, 0.0)
                    })
            
            # Sort by centrality
            bridge_details.sort(key=lambda x: x["centrality"], reverse=True)
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Bridge Node Centrality", "Bridge Node Connections"),
                specs=[[{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Bridge node centrality
            node_ids = [d["id"] for d in bridge_details]
            node_centrality = [d["centrality"] for d in bridge_details]
            node_labels = [f"{d['id']}: {d['content'][:20]}..." for d in bridge_details]
            
            fig.add_trace(
                go.Bar(
                    x=node_labels,
                    y=node_centrality,
                    marker_color="red",
                    name="Centrality"
                ),
                row=1, col=1
            )
            
            # Bridge node connections
            # Create NetworkX graph
            G: nx.Graph = nx.Graph()
            
            # Get all nodes and edges
            nodes = []
            for node in await self.graph_manager.get_concept():
                nodes.append(node)
                G.add_node(node and node.id, content=node and node.content, metadata=node and node.metadata)
            
            edges = []
            for edge in await self.graph_manager.get_relationship():
                edges.append(edge)
                G.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    metadata=edge.metadata
                )
            
            # Get communities
            communities = state.get("communities", [])
            
            # Create node-to-community mapping
            node_community = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    node_community[node_id] = i
            
            # Create subgraph with bridge nodes and their neighbors
            bridge_node_set = set(bridge_nodes)
            subgraph_nodes = set(bridge_nodes)
            
            for node_id in bridge_nodes:
                if node_id in G:
                    subgraph_nodes.update(G.neighbors(node_id))
            
            subgraph = G.subgraph(subgraph_nodes)
            
            # Use force-directed layout
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node_id in subgraph.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                # Node text
                node_content = G.nodes[node_id].get("content", "")
                node_text.append(f"{node_id}: {node_content[:30]}...")
                
                # Node size
                if node_id in bridge_node_set:
                    size = 20
                else:
                    size = 10
                node_size.append(size)
                
                # Node color
                if node_id in bridge_node_set:
                    # Bridge nodes are red
                    node_color.append("red")
                elif node_id in node_community:
                    # Color by community
                    community_idx = node_community[node_id]
                    color_idx = community_idx % len(self.color_palette)
                    node_color.append(self.color_palette[color_idx])
                else:
                    # Default color
                    node_color.append("blue")
            
            # Create edge traces
            edge_x = []
            edge_y = []
            
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Add edges
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                    name="Connections"
                ),
                row=1, col=2
            )
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers",
                    hovertext=node_text,
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=1, color="#333")
                    ),
                    name="Nodes"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                width=1200,
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
                import time
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"bridge_nodes_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"bridge_nodes_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize bridge nodes: {e}")
            return ""
    
    async def visualize_graph_evolution(
        self,
        metrics_history: List[Dict[str, Any]],
        title: str = "Graph Evolution",
        filename: Optional[str] = None
    ) -> str:
        """Visualize graph evolution over time.
        
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
            
            # Extract iterations and metrics
            iterations = [m.get("iteration", i) for i, m in enumerate(metrics_history)]
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Node and Edge Count",
                    "Modularity",
                    "Average Path Length",
                    "Bridge Node Count"
                )
            )
            
            # Node and edge count
            node_counts = [m.get("node_count", 0) for m in metrics_history]
            edge_counts = [m.get("edge_count", 0) for m in metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=node_counts,
                    mode="lines+markers",
                    name="Nodes"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=edge_counts,
                    mode="lines+markers",
                    name="Edges"
                ),
                row=1, col=1
            )
            
            # Modularity
            modularity = [m.get("modularity", 0.0) for m in metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=modularity,
                    mode="lines+markers",
                    name="Modularity"
                ),
                row=1, col=2
            )
            
            # Average path length
            avg_path_length = [m.get("avg_path_length", 0.0) for m in metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=avg_path_length,
                    mode="lines+markers",
                    name="Avg Path Length"
                ),
                row=2, col=1
            )
            
            # Bridge node count
            bridge_node_counts = [len(m.get("bridge_nodes", [])) for m in metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=bridge_node_counts,
                    mode="lines+markers",
                    name="Bridge Nodes"
                ),
                row=2, col=2
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
                height=800
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
                import time
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"evolution_{timestamp}.html")
                fig.write_html(output_path)
                
                # Also save as PNG for reports
                png_path = os.path.join(self.output_dir, f"evolution_{timestamp}.png")
                fig.write_image(png_path)
                
                return output_path
        except Exception as e:
            log.error(f"Failed to visualize graph evolution: {e}")
            return ""
