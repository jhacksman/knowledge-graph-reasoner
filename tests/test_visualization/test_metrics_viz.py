"""Tests for metrics visualization module."""
import pytest
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.visualization.metrics_viz import MetricsVisualizer


@pytest.fixture
def mock_history():
    """Create mock evaluation history."""
    return [
        {
            "iteration": 1,
            "node_count": 5,
            "edge_count": 7,
            "modularity": 0.6,
            "avg_path_length": 2.0,
            "diameter": 4,
            "community_count": 2,
            "domain_coverage": {
                "science": 0.3,
                "technology": 0.2
            }
        },
        {
            "iteration": 2,
            "node_count": 8,
            "edge_count": 12,
            "modularity": 0.7,
            "avg_path_length": 2.2,
            "diameter": 5,
            "community_count": 3,
            "domain_coverage": {
                "science": 0.35,
                "technology": 0.25
            }
        },
        {
            "iteration": 3,
            "node_count": 10,
            "edge_count": 15,
            "modularity": 0.75,
            "avg_path_length": 2.5,
            "diameter": 5,
            "community_count": 3,
            "domain_coverage": {
                "science": 0.4,
                "technology": 0.3
            }
        }
    ]


@pytest.fixture
def mock_comparison():
    """Create mock comparison data."""
    return {
        "iterations": [1, 3],
        "metrics": {
            "node_count": {
                "values": [5, 10],
                "difference": 5,
                "percent_change": 100.0
            },
            "edge_count": {
                "values": [7, 15],
                "difference": 8,
                "percent_change": 114.3
            },
            "modularity": {
                "values": [0.6, 0.75],
                "difference": 0.15,
                "percent_change": 25.0
            }
        },
        "domain_coverage": {
            "science": {
                "values": [0.3, 0.4],
                "difference": 0.1,
                "percent_change": 33.3
            },
            "technology": {
                "values": [0.2, 0.3],
                "difference": 0.1,
                "percent_change": 50.0
            }
        }
    }


def test_metrics_visualizer_initialization():
    """Test MetricsVisualizer initialization."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Check properties
        assert visualizer.output_dir == temp_dir
        
        # Check that output directory was created
        assert os.path.exists(temp_dir)


def test_visualize_metrics_over_time(mock_history):
    """Test visualizing metrics over time."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize metrics
            output_path = visualizer.visualize_metrics_over_time(
                history=mock_history,
                title="Test Metrics",
                filename="test_metrics"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_metrics" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_metrics.html")
            assert os.path.basename(output_path) == "test_metrics.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_visualize_domain_coverage(mock_history):
    """Test visualizing domain coverage."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize domain coverage
            output_path = visualizer.visualize_domain_coverage(
                history=mock_history,
                title="Test Domain Coverage",
                filename="test_domain_coverage"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_domain_coverage" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_domain_coverage.html")
            assert os.path.basename(output_path) == "test_domain_coverage.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_visualize_community_evolution(mock_history):
    """Test visualizing community evolution."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize community evolution
            output_path = visualizer.visualize_community_evolution(
                history=mock_history,
                title="Test Community Evolution",
                filename="test_community_evolution"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_community_evolution" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_community_evolution.html")
            assert os.path.basename(output_path) == "test_community_evolution.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_visualize_comparison(mock_comparison):
    """Test visualizing comparison."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize comparison
            output_path = visualizer.visualize_comparison(
                comparison=mock_comparison,
                title="Test Comparison",
                filename="test_comparison"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_comparison" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_comparison.html")
            assert os.path.basename(output_path) == "test_comparison.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_visualize_anomalies():
    """Test visualizing anomalies."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Create mock anomalies
        anomalies = [
            {
                "metric": "node_count",
                "value": 20,
                "mean": 10,
                "std": 2,
                "z_score": 5.0
            },
            {
                "metric": "modularity",
                "value": 0.3,
                "mean": 0.7,
                "std": 0.1,
                "z_score": -4.0
            }
        ]
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.write_image = MagicMock()
        
        # Mock plotly.graph_objects.Figure
        with patch('plotly.graph_objects.Figure', return_value=mock_fig):
            # Visualize anomalies
            output_path = visualizer.visualize_anomalies(
                anomalies=anomalies,
                title="Test Anomalies",
                filename="test_anomalies"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_anomalies" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_anomalies.html")
            assert os.path.basename(output_path) == "test_anomalies.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_create_dashboard(mock_history):
    """Test creating dashboard."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = MetricsVisualizer(output_dir=temp_dir)
        
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        
        # Mock plotly.graph_objects.Figure and make_subplots
        with patch('plotly.graph_objects.Figure', return_value=mock_fig), \
             patch('plotly.subplots.make_subplots', return_value=mock_fig):
            
            # Create dashboard
            output_path = visualizer.create_dashboard(
                history=mock_history,
                title="Test Dashboard",
                filename="test_dashboard"
            )
            
            # Check output path
            assert output_path is not None
            assert "test_dashboard" in output_path
            
            # Check that HTML file was created
            html_path = os.path.join(temp_dir, "test_dashboard.html")
            assert os.path.basename(output_path) == "test_dashboard.html"
            
            # Check that write_html was called
            mock_fig.write_html.assert_called_once()


def test_plot_metric_over_time(mock_history):
    """Test plotting metric over time."""
    visualizer = MetricsVisualizer(output_dir="/tmp")
    
    # Mock plotly figure
    mock_fig = MagicMock()
    
    # Plot metric
    fig = visualizer._plot_metric_over_time(
        history=mock_history,
        metric="node_count",
        title="Node Count Over Time",
        fig=mock_fig
    )
    
    # Check result
    assert fig is not None
    assert fig == mock_fig


def test_plot_domain_coverage(mock_history):
    """Test plotting domain coverage."""
    visualizer = MetricsVisualizer(output_dir="/tmp")
    
    # Mock plotly figure
    mock_fig = MagicMock()
    
    # Plot domain coverage
    fig = visualizer._plot_domain_coverage(
        history=mock_history,
        title="Domain Coverage",
        fig=mock_fig
    )
    
    # Check result
    assert fig is not None
    assert fig == mock_fig


def test_plot_comparison_metrics(mock_comparison):
    """Test plotting comparison metrics."""
    visualizer = MetricsVisualizer(output_dir="/tmp")
    
    # Mock plotly figure
    mock_fig = MagicMock()
    
    # Plot comparison
    fig = visualizer._plot_comparison_metrics(
        comparison=mock_comparison,
        title="Metrics Comparison",
        fig=mock_fig
    )
    
    # Check result
    assert fig is not None
    assert fig == mock_fig
