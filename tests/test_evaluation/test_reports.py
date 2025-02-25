"""Tests for evaluation reports module."""
import pytest
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.reports import ReportGenerator
from src.models.node import Node
from src.models.edge import Edge


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
                metadata={"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
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


@pytest.fixture
def mock_evaluator():
    """Create a mock graph evaluator."""
    evaluator = AsyncMock()
    
    # Mock evaluate
    async def mock_evaluate(iteration=None):
        return {
            "iteration": iteration or 0,
            "node_count": 10,
            "edge_count": 15,
            "modularity": 0.75,
            "avg_path_length": 2.5,
            "diameter": 5,
            "community_count": 3,
            "domain_coverage": {
                "science": 0.4,
                "technology": 0.3,
                "humanities": 0.2
            }
        }
    
    evaluator.evaluate = mock_evaluate
    
    # Mock get_history
    async def mock_get_history():
        return [
            {
                "iteration": 1,
                "node_count": 5,
                "edge_count": 7,
                "modularity": 0.6,
                "avg_path_length": 2.0,
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
                "domain_coverage": {
                    "science": 0.4,
                    "technology": 0.3
                }
            }
        ]
    
    evaluator.get_history = mock_get_history
    
    # Mock compare_iterations
    async def mock_compare_iterations(iter1, iter2):
        return {
            "iterations": [iter1, iter2],
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
            },
            "node_overlap": {
                "common_count": 5,
                "reference_only": 0,
                "current_only": 5,
                "overlap_percentage": 50.0
            },
            "edge_overlap": {
                "common_count": 7,
                "reference_only": 0,
                "current_only": 8,
                "overlap_percentage": 46.7
            }
        }
    
    evaluator.compare_iterations = mock_compare_iterations
    
    return evaluator


@pytest.mark.asyncio
async def test_report_generator_initialization(mock_graph_manager, mock_evaluator):
    """Test ReportGenerator initialization."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Check properties
        assert generator.graph_manager == mock_graph_manager
        assert generator.evaluator == mock_evaluator
        assert generator.output_dir == temp_dir
        
        # Check that output directory was created
        assert os.path.exists(temp_dir)
        
        # Check that Jinja environment was initialized
        assert generator.jinja_env is not None


@pytest.mark.asyncio
async def test_generate_markdown_report(mock_graph_manager, mock_evaluator):
    """Test generating Markdown report."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Mock visualization methods
        with patch('src.visualization.graph_viz.GraphVisualizer.visualize_graph', 
                  return_value=os.path.join(temp_dir, "visualizations/graph.png")), \
             patch('src.visualization.graph_viz.GraphVisualizer.visualize_communities',
                  return_value=os.path.join(temp_dir, "visualizations/communities.png")), \
             patch('src.visualization.metrics_viz.MetricsVisualizer.visualize_metrics_over_time',
                  return_value=os.path.join(temp_dir, "visualizations/metrics.png")):
            
            # Generate report
            report_path = await generator.generate_markdown_report(
                title="Test Report",
                include_visualizations=True,
                filename="test_report"
            )
            
            # Check that report was created
            assert os.path.exists(report_path)
            assert os.path.basename(report_path) == "test_report.md"
            
            # Check report content
            with open(report_path, "r") as f:
                content = f.read()
                
                # Check that title is included
                assert "Test Report" in content
                
                # Check that metrics are included
                assert "Nodes: 10" in content
                assert "Edges: 15" in content
                assert "Modularity: 0.75" in content
                
                # Check that visualization paths are included
                assert "visualizations/graph.png" in content or "visualizations/communities.png" in content


@pytest.mark.asyncio
async def test_generate_html_report(mock_graph_manager, mock_evaluator):
    """Test generating HTML report."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Mock visualization methods and markdown generation
        with patch('src.visualization.graph_viz.GraphVisualizer.visualize_graph', 
                  return_value=os.path.join(temp_dir, "visualizations/graph.png")), \
             patch('src.visualization.graph_viz.GraphVisualizer.visualize_communities',
                  return_value=os.path.join(temp_dir, "visualizations/communities.png")), \
             patch('src.visualization.metrics_viz.MetricsVisualizer.visualize_metrics_over_time',
                  return_value=os.path.join(temp_dir, "visualizations/metrics.png")):
            
            # Generate report
            report_path = await generator.generate_html_report(
                title="Test HTML Report",
                include_visualizations=True,
                filename="test_html_report"
            )
            
            # Check that report was created
            assert os.path.exists(report_path)
            assert os.path.basename(report_path) == "test_html_report.html"
            
            # Check report content
            with open(report_path, "r") as f:
                content = f.read()
                
                # Check that title is included
                assert "Test HTML Report" in content
                
                # Check that HTML styling is included
                assert "<style>" in content
                assert "</style>" in content
                
                # Check that HTML structure is correct
                assert "<!DOCTYPE html>" in content
                assert "<html>" in content
                assert "</html>" in content


@pytest.mark.asyncio
async def test_generate_comparison_report(mock_graph_manager, mock_evaluator):
    """Test generating comparison report."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Create template directory and file
        template_dir = os.path.join(temp_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        comparison_template = """# Comparison Report: {{ report_title }}
## Comparison Summary
{{ comparison_summary }}
## Metrics Comparison
{% for metric_name, metric_data in comparison.metrics.items() %}
{{ metric_name }}: {{ metric_data.values[0] }} -> {{ metric_data.values[1] }}
{% endfor %}
"""
        
        with open(os.path.join(template_dir, "comparison_template.md"), "w") as f:
            f.write(comparison_template)
        
        # Set template directory
        generator.template_dir = template_dir
        generator.jinja_env = generator.jinja_env.overlay(
            loader=generator.jinja_env.loader.overlay(
                generator.jinja_env.loader.__class__([template_dir])
            )
        )
        
        # Mock visualization method
        with patch('src.visualization.metrics_viz.MetricsVisualizer.visualize_comparison',
                  return_value=os.path.join(temp_dir, "visualizations/comparison.png")):
            
            # Generate report
            report_path = await generator.generate_comparison_report(
                iter1=1,
                iter2=3,
                title="Test Comparison",
                filename="test_comparison"
            )
            
            # Check that report was created
            assert os.path.exists(report_path)
            assert os.path.basename(report_path) == "test_comparison.md"
            
            # Check report content
            with open(report_path, "r") as f:
                content = f.read()
                
                # Check that title is included
                assert "Test Comparison" in content
                
                # Check that metrics are included
                assert "node_count" in content
                assert "edge_count" in content
                assert "modularity" in content


@pytest.mark.asyncio
async def test_generate_dashboard(mock_graph_manager, mock_evaluator):
    """Test generating dashboard."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Mock dashboard creation
        dashboard_path = os.path.join(temp_dir, "dashboard/dashboard.html")
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        
        with open(dashboard_path, "w") as f:
            f.write("<html><body>Test Dashboard</body></html>")
        
        with patch('src.visualization.metrics_viz.MetricsVisualizer.create_dashboard',
                  return_value=dashboard_path):
            
            # Generate dashboard
            result_path = await generator.generate_dashboard(
                title="Test Dashboard",
                filename="test_dashboard"
            )
            
            # Check result
            assert result_path == dashboard_path
            assert os.path.exists(result_path)


@pytest.mark.asyncio
async def test_executive_summary_generation(mock_graph_manager, mock_evaluator):
    """Test executive summary generation."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = ReportGenerator(
            graph_manager=mock_graph_manager,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Test data
        data = {
            "node_count": 10,
            "edge_count": 15,
            "modularity": 0.75,
            "bridge_node_count": 2,
            "metrics": {
                "community_count": 3,
                "domain_coverage": {
                    "science": 0.4,
                    "technology": 0.3,
                    "humanities": 0.2
                }
            },
            "history": [
                {
                    "node_count": 5,
                    "edge_count": 7
                },
                {
                    "node_count": 10,
                    "edge_count": 15
                }
            ]
        }
        
        # Generate summary
        summary = generator._generate_executive_summary(data)
        
        # Check content
        assert "10 concepts" in summary
        assert "15 relationships" in summary
        assert "0.75" in summary
        assert "3 distinct communities" in summary
        assert "2 bridge nodes" in summary
        assert "grown by 5 nodes" in summary
