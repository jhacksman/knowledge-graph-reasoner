"""Tests for graph visualization module."""
import pytest
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import networkx as nx
import numpy as np

from src.models.node import Node
from src.models.edge import Edge
from src.visualization.graph_viz import GraphVisualizer


@pytest.fixture
def mock_graph_manager():
    """Create a mock graph manager."""
    manager = AsyncMock()
    
    # Mock get_graph_state
    manager.get_graph_state.return_value = {
        "timestamp": 1234567890,
        "node_count": 10,
        "edge_count": 15,
        "modularity": 0.75,
        "avg_path_length": 2.5,
        "diameter": 5,
        "communities": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9", "10"]],
        "bridge_nodes": ["3", "6"],
        "centrality": {
            "1": 0.1, "2": 0.2, "3": 0.8, "4": 0.3, 
            "5": 0.2, "6": 0.7, "7": 0.4, "8": 0.3,
            "9": 0.2, "10": 0.1
        }
    }
    
    # Mock get_all_concepts
    async def mock_get_all_concepts():
        for i in range(1, 11):
            yield Node(
                id=str(i),
                content=f"Content for node {i}",
                metadata={"embedding": np.random.rand(10).tolist()}
            )
    
    manager.get_all_concepts = mock_get_all_concepts
    
    # Mock get_all_relationships
    async def mock_get_all_relationships():
        edges = [
            ("1", "2", "related_to"),
            ("2", "3", "causes"),
            ("3", "4", "part_of"),
            ("4", "5", "related_to"),
            ("5", "6", "causes")
        ]
        
        for source, target, rel_type in edges:
            yield Edge(
                source=source,
                target=target,
                type=rel_type,
                metadata={"confidence": 0.9}
            )
    
    manager.get_all_relationships = mock_get_all_relationships
    
    return manager


@pytest.mark.asyncio
async def test_graph_visualizer_initialization(mock_graph_manager):
    """Test GraphVisualizer initialization."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Check properties
        assert visualizer.graph_manager == mock_graph_manager
        assert visualizer.output_dir == temp_dir
        
        # Check that output directory was created
        assert os.path.exists(temp_dir)


@pytest.mark.asyncio
async def test_build_networkx_graph(mock_graph_manager):
    """Test building NetworkX graph."""
    visualizer = GraphVisualizer(
        graph_manager=mock_graph_manager,
        output_dir="/tmp"
    )
    
    # Build graph
    graph = await visualizer.build_networkx_graph()
    
    # Check graph
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() > 0
    
    # Check node attributes
    for node in graph.nodes():
        assert "content" in graph.nodes[node]
        assert "centrality" in graph.nodes[node]
    
    # Check edge attributes
    for u, v in graph.edges():
        assert "type" in graph.edges[u, v]
        assert "confidence" in graph.edges[u, v]


@pytest.mark.asyncio
async def test_visualize_graph(mock_graph_manager):
    """Test visualizing graph."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize graph
            output_path = await visualizer.visualize_graph(
                title="Test Graph",
                filename="test_graph"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_graph" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_graph.html")
            assert os.path.basename(output_path) == "test_graph.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


@pytest.mark.asyncio
async def test_visualize_communities(mock_graph_manager):
    """Test visualizing communities."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize communities
            output_path = await visualizer.visualize_communities(
                title="Test Communities",
                filename="test_communities"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_communities" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_communities.html")
            assert os.path.basename(output_path) == "test_communities.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


@pytest.mark.asyncio
async def test_visualize_bridge_nodes(mock_graph_manager):
    """Test visualizing bridge nodes."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize bridge nodes
            output_path = await visualizer.visualize_bridge_nodes(
                title="Test Bridge Nodes",
                filename="test_bridge_nodes"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_bridge_nodes" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_bridge_nodes.html")
            assert os.path.basename(output_path) == "test_bridge_nodes.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


@pytest.mark.asyncio
async def test_visualize_graph_evolution(mock_graph_manager):
    """Test visualizing graph evolution."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Create mock history
        history = [
            {
                "iteration": 1,
                "graph": nx.Graph(),
                "communities": [["1", "2"], ["3", "4"]],
                "bridge_nodes": ["2"]
            },
            {
                "iteration": 2,
                "graph": nx.Graph(),
                "communities": [["1", "2", "5"], ["3", "4", "6"]],
                "bridge_nodes": ["2", "5"]
            }
        ]
        
        # Add nodes and edges to graphs
        for h in history:
            g = h["graph"]
            for i in range(1, 7):
                g.add_node(str(i), content=f"Content {i}")
            
            g.add_edge("1", "2", type="related_to")
            g.add_edge("3", "4", type="related_to")
            
            if h["iteration"] == 2:
                g.add_edge("2", "5", type="related_to")
                g.add_edge("4", "6", type="related_to")
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize graph evolution
            output_path = visualizer.visualize_graph_evolution(
                history=history,
                title="Test Evolution",
                filename="test_evolution"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_evolution" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_evolution.html")
            assert os.path.basename(output_path) == "test_evolution.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


@pytest.mark.asyncio
async def test_create_interactive_visualization(mock_graph_manager):
    """Test creating interactive visualization."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = GraphVisualizer(
            graph_manager=mock_graph_manager,
            output_dir=temp_dir
        )
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Build graph
            graph = await visualizer.build_networkx_graph()
            
            # Create interactive visualization
            output_path = visualizer.create_interactive_visualization(
                graph=graph,
                title="Test Interactive",
                filename="test_interactive",
                node_color_by="community",
                node_size_by="centrality"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_interactive" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_interactive.html")
            assert os.path.basename(output_path) == "test_interactive.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()
