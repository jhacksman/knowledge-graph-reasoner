"""Tests for evaluation benchmarks module."""
import pytest
import os
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.benchmarks import GraphBenchmark, CustomBenchmark
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
                metadata={}
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
                metadata={}
            )
    
    manager.get_all_relationships = mock_get_all_relationships
    
    return manager


@pytest.fixture
def mock_pipeline():
    """Create a mock reasoning pipeline."""
    pipeline = AsyncMock()
    
    # Mock run
    async def mock_run(seed_concepts, max_iterations=10):
        return {
            "iterations": max_iterations,
            "node_count": 20,
            "edge_count": 30,
            "modularity": 0.8
        }
    
    pipeline.run = mock_run
    
    return pipeline


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
            }
        }
    
    evaluator.compare_iterations = mock_compare_iterations
    
    return evaluator


@pytest.mark.asyncio
async def test_graph_benchmark_initialization():
    """Test GraphBenchmark initialization."""
    benchmark = GraphBenchmark(
        graph_manager=AsyncMock(),
        pipeline=AsyncMock(),
        evaluator=AsyncMock(),
        output_dir="/tmp/benchmarks"
    )
    
    # Check standard seeds
    assert "science" in benchmark.standard_seeds
    assert "technology" in benchmark.standard_seeds
    assert "humanities" in benchmark.standard_seeds
    assert "interdisciplinary" in benchmark.standard_seeds
    
    # Check output directory
    assert benchmark.output_dir == "/tmp/benchmarks"


@pytest.mark.asyncio
async def test_run_benchmark(mock_graph_manager, mock_pipeline, mock_evaluator):
    """Test running a benchmark."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=mock_pipeline,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Run benchmark
        result = await benchmark.run_benchmark(
            seed_concepts=["quantum mechanics", "artificial intelligence"],
            max_iterations=5,
            name="test_benchmark"
        )
        
        # Check result
        assert result is not None
        assert "name" in result
        assert result["name"] == "test_benchmark"
        assert "seed_concepts" in result
        assert "iterations" in result
        assert result["iterations"] == 5
        assert "metrics" in result
        assert "timestamp" in result
        
        # Check that result was saved
        result_files = os.listdir(os.path.join(temp_dir, "results"))
        assert len(result_files) > 0


@pytest.mark.asyncio
async def test_create_reference_graph(mock_graph_manager, mock_evaluator):
    """Test creating a reference graph."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=AsyncMock(),
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Create reference graph
        ref_name = await benchmark.create_reference_graph(
            name="test_reference",
            description="Test reference graph"
        )
        
        # Check result
        assert ref_name is not None
        assert ref_name == "test_reference"
        
        # Check that reference was saved
        ref_dir = os.path.join(temp_dir, "references")
        assert os.path.exists(ref_dir)
        ref_files = os.listdir(ref_dir)
        assert len(ref_files) > 0
        
        # Check reference content
        ref_file = os.path.join(ref_dir, f"{ref_name}.json")
        assert os.path.exists(ref_file)
        
        with open(ref_file, "r") as f:
            ref_data = json.load(f)
        
        assert "name" in ref_data
        assert ref_data["name"] == "test_reference"
        assert "description" in ref_data
        assert "timestamp" in ref_data
        assert "graph_state" in ref_data
        assert "metrics" in ref_data


@pytest.mark.asyncio
async def test_compare_with_reference(mock_graph_manager, mock_evaluator):
    """Test comparing with a reference graph."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=AsyncMock(),
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Create reference directory and file
        ref_dir = os.path.join(temp_dir, "references")
        os.makedirs(ref_dir, exist_ok=True)
        
        ref_data = {
            "name": "test_reference",
            "description": "Test reference graph",
            "timestamp": 1234567890,
            "graph_state": {
                "node_count": 5,
                "edge_count": 7,
                "modularity": 0.6
            },
            "metrics": {
                "node_count": 5,
                "edge_count": 7,
                "modularity": 0.6,
                "avg_path_length": 2.0,
                "domain_coverage": {
                    "science": 0.3,
                    "technology": 0.2
                }
            }
        }
        
        ref_file = os.path.join(ref_dir, "test_reference.json")
        with open(ref_file, "w") as f:
            json.dump(ref_data, f)
        
        # Compare with reference
        comparison = await benchmark.compare_with_reference("test_reference")
        
        # Check result
        assert comparison is not None
        assert "reference" in comparison
        assert comparison["reference"] == "test_reference"
        assert "metrics" in comparison
        assert "node_count" in comparison["metrics"]
        assert "edge_count" in comparison["metrics"]
        assert "modularity" in comparison["metrics"]
        assert "domain_coverage" in comparison["domain_coverage"]


@pytest.mark.asyncio
async def test_list_reference_graphs(mock_graph_manager):
    """Test listing reference graphs."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=AsyncMock(),
            evaluator=AsyncMock(),
            output_dir=temp_dir
        )
        
        # Create reference directory and files
        ref_dir = os.path.join(temp_dir, "references")
        os.makedirs(ref_dir, exist_ok=True)
        
        ref_data = {
            "name": "test_reference",
            "description": "Test reference graph",
            "timestamp": 1234567890
        }
        
        # Create multiple reference files
        for i in range(3):
            ref_file = os.path.join(ref_dir, f"reference_{i}.json")
            with open(ref_file, "w") as f:
                json.dump(ref_data, f)
        
        # List reference graphs
        references = benchmark.list_reference_graphs()
        
        # Check result
        assert references is not None
        assert len(references) == 3


@pytest.mark.asyncio
async def test_list_benchmark_results(mock_graph_manager):
    """Test listing benchmark results."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=AsyncMock(),
            evaluator=AsyncMock(),
            output_dir=temp_dir
        )
        
        # Create results directory and files
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        result_data = {
            "name": "test_benchmark",
            "seed_concepts": ["concept1", "concept2"],
            "timestamp": 1234567890
        }
        
        # Create multiple result files
        for i in range(3):
            result_file = os.path.join(results_dir, f"benchmark_{i}.json")
            with open(result_file, "w") as f:
                json.dump(result_data, f)
        
        # List benchmark results
        results = benchmark.list_benchmark_results()
        
        # Check result
        assert results is not None
        assert len(results) == 3


@pytest.mark.asyncio
async def test_run_standard_benchmark(mock_graph_manager, mock_pipeline, mock_evaluator):
    """Test running a standard benchmark."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = GraphBenchmark(
            graph_manager=mock_graph_manager,
            pipeline=mock_pipeline,
            evaluator=mock_evaluator,
            output_dir=temp_dir
        )
        
        # Run standard benchmark
        result = await benchmark.run_standard_benchmark(
            domain="science",
            max_iterations=3
        )
        
        # Check result
        assert result is not None
        assert "name" in result
        assert "science" in result["name"]
        assert "seed_concepts" in result
        assert "iterations" in result
        assert result["iterations"] == 3
        assert "metrics" in result


@pytest.mark.asyncio
async def test_custom_benchmark():
    """Test CustomBenchmark class."""
    # Create custom benchmark
    custom = CustomBenchmark(
        name="Custom Test",
        seed_concepts=["concept1", "concept2"],
        max_iterations=5,
        target_metrics={
            "modularity": 0.7,
            "avg_path_length": 3.0
        }
    )
    
    # Check properties
    assert custom.name == "Custom Test"
    assert custom.seed_concepts == ["concept1", "concept2"]
    assert custom.max_iterations == 5
    assert "modularity" in custom.target_metrics
    assert custom.target_metrics["modularity"] == 0.7
    
    # Test evaluate_results
    results = {
        "metrics": {
            "modularity": 0.8,
            "avg_path_length": 2.5
        }
    }
    
    evaluation = custom.evaluate_results(results)
    
    # Check evaluation
    assert evaluation is not None
    assert "success" in evaluation
    assert evaluation["success"] is True
    assert "metrics" in evaluation
    assert "modularity" in evaluation["metrics"]
    assert evaluation["metrics"]["modularity"]["target"] == 0.7
    assert evaluation["metrics"]["modularity"]["actual"] == 0.8
    assert evaluation["metrics"]["modularity"]["achieved"] is True
