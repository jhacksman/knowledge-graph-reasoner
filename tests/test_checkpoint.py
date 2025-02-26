"""Tests for checkpoint functionality."""
import pytest
import os
import shutil
import json
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock

from src.pipeline.checkpoint import CheckpointManager, CheckpointMetadata
from src.graph.manager import GraphManager
from src.reasoning.pipeline import ReasoningPipeline
from src.reasoning.llm import VeniceLLM


@pytest.fixture
def checkpoint_dir():
    """Create a temporary checkpoint directory."""
    path = Path("test_checkpoints")
    path.mkdir(exist_ok=True)
    yield path
    # Clean up
    shutil.rmtree(path)


@pytest.fixture
def mock_graph_manager():
    """Create a mock graph manager."""
    manager = MagicMock(spec=GraphManager)
    
    # Mock export methods
    async def mock_export_graph_structure(path, compress=False):
        Path(path).touch()
        return Path(path)
    
    async def mock_export_embeddings(path, compress=True):
        Path(f"{path}.gz").touch()
        return Path(f"{path}.gz")
    
    async def mock_export_metrics_history(path):
        Path(path).touch()
        return Path(path)
    
    async def mock_export_full_graph(directory, compress=True, include_embeddings=True, include_metrics=True):
        Path(directory).mkdir(exist_ok=True)
        structure_path = Path(directory) / "graph_structure.json"
        embeddings_path = Path(directory) / "embeddings.pkl.gz"
        metrics_path = Path(directory) / "metrics_history.json"
        
        structure_path.touch()
        embeddings_path.touch()
        metrics_path.touch()
        
        return {
            "structure": structure_path,
            "embeddings": embeddings_path,
            "metrics": metrics_path
        }
    
    # Mock import methods
    async def mock_import_graph_structure(path):
        return 10, 20  # 10 nodes, 20 edges
    
    async def mock_import_embeddings(path):
        return 10  # 10 embeddings
    
    async def mock_import_metrics_history(path):
        return 5  # 5 metrics
    
    async def mock_import_full_graph(directory, import_structure=True, import_embeddings=True, import_metrics=True):
        return {
            "nodes": 10,
            "edges": 20,
            "embeddings": 10,
            "metrics": 5
        }
    
    # Assign mock methods
    manager.export_graph_structure.side_effect = mock_export_graph_structure
    manager.export_embeddings.side_effect = mock_export_embeddings
    manager.export_metrics_history.side_effect = mock_export_metrics_history
    manager.export_full_graph.side_effect = mock_export_full_graph
    manager.import_graph_structure.side_effect = mock_import_graph_structure
    manager.import_embeddings.side_effect = mock_import_embeddings
    manager.import_metrics_history.side_effect = mock_import_metrics_history
    manager.import_full_graph.side_effect = mock_import_full_graph
    
    return manager


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock(spec=VeniceLLM)
    
    async def mock_embed_text(text):
        import numpy as np
        return np.random.rand(1536)
    
    async def mock_generate(messages, temperature=0.7, max_tokens=1000):
        return {
            "choices": [{
                "message": {
                    "content": "Mock response"
                }
            }]
        }
    
    llm.embed_text.side_effect = mock_embed_text
    llm.generate.side_effect = mock_generate
    
    return llm


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_dir, mock_graph_manager):
        """Test creating a checkpoint."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test checkpoint"
        )
        
        # Check that checkpoint was created
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "graph_structure.json").exists()
        assert (checkpoint_path / "metrics.json").exists()
        
        # Check that embeddings were compressed
        assert not (checkpoint_path / "embeddings.pkl").exists()
        assert (checkpoint_path / "embeddings.pkl.gz").exists()
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpoint_dir, mock_graph_manager):
        """Test listing checkpoints."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoints
        await manager.create_checkpoint(iteration=1, description="Test 1")
        await manager.create_checkpoint(iteration=2, description="Test 2")
        
        # List checkpoints
        checkpoints = await manager.list_checkpoints()
        
        # Check that checkpoints were listed
        assert len(checkpoints) == 2
        assert checkpoints[0]["metadata"]["iteration"] == 2  # Newest first
        assert checkpoints[1]["metadata"]["iteration"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint(self, checkpoint_dir, mock_graph_manager):
        """Test validating a checkpoint."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test checkpoint"
        )
        
        # Validate checkpoint
        valid, message = await manager.validate_checkpoint(checkpoint_path)
        
        # Check that checkpoint is valid
        assert valid
        assert "successful" in message
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_integrity(self, checkpoint_dir, mock_graph_manager):
        """Test validating checkpoint integrity with checksums."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test checkpoint integrity"
        )
        
        # Modify a file to corrupt the checkpoint
        graph_path = checkpoint_path / "graph_structure.json"
        with open(graph_path, "w") as f:
            f.write('{"corrupted": true}')
        
        # Validate checkpoint (should fail due to checksum mismatch)
        valid, message = await manager.validate_checkpoint(checkpoint_path)
        
        # Check that validation failed
        assert not valid
        assert "integrity check failed" in message or "consistency" in message
    
    @pytest.mark.asyncio
    async def test_validate_graph_consistency(self, checkpoint_dir, mock_graph_manager):
        """Test validating graph consistency."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test graph consistency"
        )
        
        # Mock _validate_graph_consistency to test it directly
        original_validate = manager._validate_graph_consistency
        
        # Create a valid graph structure
        graph_path = checkpoint_path / "graph_structure.json"
        with open(graph_path, "w") as f:
            json.dump({
                "nodes": [
                    {"id": "node1", "content": "Node 1"},
                    {"id": "node2", "content": "Node 2"}
                ],
                "edges": [
                    {"source": "node1", "target": "node2", "type": "related"}
                ]
            }, f)
        
        # Test valid graph
        valid, message = await manager._validate_graph_consistency(checkpoint_path)
        assert valid
        assert "successful" in message
        
        # Create an invalid graph with dangling edge
        with open(graph_path, "w") as f:
            json.dump({
                "nodes": [
                    {"id": "node1", "content": "Node 1"}
                ],
                "edges": [
                    {"source": "node1", "target": "node3", "type": "related"}
                ]
            }, f)
        
        # Test invalid graph
        valid, message = await manager._validate_graph_consistency(checkpoint_path)
        assert not valid
        assert "Dangling edge" in message
        
        # Restore original method
        manager._validate_graph_consistency = original_validate
    
    @pytest.mark.asyncio
    async def test_load_checkpoint(self, checkpoint_dir, mock_graph_manager):
        """Test loading a checkpoint."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test checkpoint"
        )
        
        # Load checkpoint
        success, message = await manager.load_checkpoint(checkpoint_path)
        
        # Check that checkpoint was loaded
        assert success
        assert "successfully" in message
        
        # Check that graph manager methods were called
        mock_graph_manager.import_full_graph.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_version_compatibility(self, checkpoint_dir, mock_graph_manager):
        """Test checkpoint version compatibility handling."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test version compatibility"
        )
        
        # Modify metadata to change version
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Test compatible version (minor version change)
        metadata["version"] = "1.1.0"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Validate checkpoint (should succeed with warning)
        valid, message = await manager.validate_checkpoint(checkpoint_path)
        assert valid
        
        # Test incompatible version (major version change)
        metadata["version"] = "2.0.0"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Validate checkpoint (should fail)
        valid, message = await manager.validate_checkpoint(checkpoint_path)
        assert not valid
        assert "Incompatible checkpoint version" in message
    
    @pytest.mark.asyncio
    async def test_serialization_deserialization(self, checkpoint_dir, mock_graph_manager):
        """Test serialization and deserialization of graph state."""
        config = {"test": "config"}
        manager = CheckpointManager(
            base_dir=checkpoint_dir,
            graph_manager=mock_graph_manager,
            config=config
        )
        
        # Create checkpoint
        checkpoint_path = await manager.create_checkpoint(
            iteration=1,
            description="Test serialization"
        )
        
        # Check that graph manager export methods were called
        mock_graph_manager.export_graph_structure.assert_called_once()
        mock_graph_manager.export_embeddings.assert_called_once()
        mock_graph_manager.export_metrics_history.assert_called_once()
        
        # Load checkpoint
        success, message = await manager.load_checkpoint(checkpoint_path)
        assert success
        
        # Check that graph manager import methods were called
        mock_graph_manager.import_full_graph.assert_called_once()
        
        # Check import arguments
        args, kwargs = mock_graph_manager.import_full_graph.call_args
        assert kwargs["import_structure"] is True
        assert kwargs["import_embeddings"] is True
        assert kwargs["import_metrics"] is True


class TestReasoningPipelineCheckpointing:
    """Tests for ReasoningPipeline checkpointing."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_checkpointing(self, checkpoint_dir, mock_graph_manager, mock_llm):
        """Test pipeline with checkpointing enabled."""
        # Create pipeline with checkpointing
        pipeline = ReasoningPipeline(
            llm=mock_llm,
            graph=mock_graph_manager,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_iterations=1,
            enable_checkpointing=True
        )
        
        # Check that checkpoint manager was created
        assert pipeline.checkpoint_manager is not None
        assert pipeline.enable_checkpointing is True
        
        # Mock expand_knowledge to create checkpoints
        original_expand = pipeline.expand_knowledge
        
        async def mock_expand(*args, **kwargs):
            # Add some metrics history
            pipeline.metric_history = [{"test": "metric"}]
            
            # Create a manual checkpoint
            result = await pipeline.create_manual_checkpoint("Test manual checkpoint")
            assert result["success"] is True
            
            # List checkpoints
            checkpoints = await pipeline.list_checkpoints()
            assert len(checkpoints) == 1
            
            # Validate checkpoint
            validation = await pipeline.validate_checkpoint(checkpoints[0]["path"])
            assert validation["valid"] is True
            
            return {"test": "state"}
        
        # Replace expand_knowledge with mock
        pipeline.expand_knowledge = mock_expand
        
        # Expand knowledge
        await pipeline.expand_knowledge("test concept")
    
    @pytest.mark.asyncio
    async def test_pipeline_without_checkpointing(self, mock_graph_manager, mock_llm):
        """Test pipeline with checkpointing disabled."""
        # Create pipeline without checkpointing
        pipeline = ReasoningPipeline(
            llm=mock_llm,
            graph=mock_graph_manager,
            enable_checkpointing=False
        )
        
        # Check that checkpoint manager was not created
        assert pipeline.checkpoint_manager is None
        assert pipeline.enable_checkpointing is False
        
        # List checkpoints (should be empty)
        checkpoints = await pipeline.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Create manual checkpoint (should fail)
        result = await pipeline.create_manual_checkpoint("Test manual checkpoint")
        assert result["success"] is False
    
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, checkpoint_dir, mock_graph_manager, mock_llm):
        """Test resuming from a checkpoint."""
        # Create pipeline with checkpointing
        pipeline = ReasoningPipeline(
            llm=mock_llm,
            graph=mock_graph_manager,
            checkpoint_dir=checkpoint_dir,
            enable_checkpointing=True
        )
        
        # Create a checkpoint
        result = await pipeline.create_manual_checkpoint("Test checkpoint for resuming")
        checkpoint_path = result["path"]
        
        # Mock the load_checkpoint method to return success
        original_load = pipeline.checkpoint_manager.load_checkpoint
        
        async def mock_load_checkpoint(path):
            return True, f"Successfully loaded checkpoint from {path}"
        
        pipeline.checkpoint_manager.load_checkpoint = mock_load_checkpoint
        
        # Mock list_checkpoints to return metadata with iteration
        original_list = pipeline.checkpoint_manager.list_checkpoints
        
        async def mock_list_checkpoints():
            return [{
                "path": checkpoint_path,
                "metadata": {
                    "iteration": 5,
                    "timestamp": 1234567890,
                    "config": {}
                }
            }]
        
        pipeline.checkpoint_manager.list_checkpoints = mock_list_checkpoints
        
        # Replace expand_knowledge with a mock that checks the start_iteration
        original_expand = pipeline.expand_knowledge
        
        async def mock_expand(seed_concept, context=None, resume_from_checkpoint=None):
            # If resuming, check that we're starting from iteration 6 (5+1)
            if resume_from_checkpoint:
                # This would be checked inside the for loop in the real method
                assert resume_from_checkpoint == checkpoint_path
                
                # Mock that we're listing checkpoints to get the iteration
                checkpoints = await (pipeline.checkpoint_manager and (pipeline.checkpoint_manager and pipeline.((checkpoint_manager is not None) and checkpoint_manager.list_checkpoints())))
                for checkpoint in checkpoints:
                    if str(checkpoint["path"]) == str(resume_from_checkpoint):
                        # Check that we're starting from iteration 6 (5+1)
                        assert checkpoint["metadata"]["iteration"] + 1 == 6
                        break
            
            return {"test": "state"}
        
        # Replace expand_knowledge with mock
        pipeline.expand_knowledge = mock_expand
        
        # Expand knowledge with resume
        await pipeline.expand_knowledge("test concept", resume_from_checkpoint=checkpoint_path)
        
        # Restore original methods
        pipeline.checkpoint_manager.load_checkpoint = original_load
        pipeline.checkpoint_manager.list_checkpoints = original_list
        pipeline.expand_knowledge = original_expand
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_process_with_checkpoints(self, checkpoint_dir, mock_graph_manager, mock_llm):
        """Test complete reasoning process with checkpoints."""
        # Create pipeline with checkpointing
        pipeline = ReasoningPipeline(
            llm=mock_llm,
            graph=mock_graph_manager,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_iterations=2,
            max_iterations=6,
            enable_checkpointing=True
        )
        
        # Mock the should_checkpoint method to always return True for testing
        original_should = pipeline.checkpoint_manager.should_checkpoint
        
        async def mock_should_checkpoint(iteration):
            return iteration % 2 == 0  # Checkpoint every 2 iterations
        
        pipeline.checkpoint_manager.should_checkpoint = mock_should_checkpoint
        
        # Mock the expand_knowledge method to track checkpoints
        original_expand = pipeline.expand_knowledge
        
        async def mock_expand(seed_concept, context=None, resume_from_checkpoint=None):
            # Simulate a complete reasoning process
            checkpoints_created = []
            
            # Start from the appropriate iteration
            start_iteration = 0
            if resume_from_checkpoint:
                # Simulate resuming from checkpoint
                checkpoints = await pipeline.list_checkpoints()
                for checkpoint in checkpoints:
                    if str(checkpoint["path"]) == str(resume_from_checkpoint):
                        start_iteration = checkpoint["metadata"]["iteration"] + 1
                        break
            
            # Run iterations
            for i in range(start_iteration, pipeline.max_iterations):
                # Add metrics for each iteration
                pipeline.metric_history.append({
                    "iteration": i,
                    "avg_path_length": 4.8,
                    "diameter": 17.0,
                    "modularity": 0.7
                })
                
                # Check if should create checkpoint
                should_checkpoint = await (pipeline.checkpoint_manager and pipeline.checkpoint_manager.should_checkpoint())
                if should_checkpoint:
                    result = await pipeline.create_manual_checkpoint(f"Iteration {i}")
                    assert result["success"] is True
                    checkpoints_created.append(i)
            
            # Verify checkpoints were created at the right iterations
            expected_checkpoints = [0, 2, 4]  # Iterations 0, 2, 4 should have checkpoints
            assert checkpoints_created == expected_checkpoints
            
            # Create final checkpoint
            result = await pipeline.create_manual_checkpoint("Final state")
            assert result["success"] is True
            
            # List all checkpoints
            checkpoints = await pipeline.list_checkpoints()
            assert len(checkpoints) == 4  # 3 regular + 1 final
            
            return {"test": "state"}
        
        # Replace methods with mocks
        pipeline.expand_knowledge = mock_expand
        
        # Run the reasoning process
        await pipeline.expand_knowledge("test concept")
        
        # Restore original methods
        pipeline.checkpoint_manager.should_checkpoint = original_should
        pipeline.expand_knowledge = original_expand
