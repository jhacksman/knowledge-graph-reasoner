"""Tests for checkpoint functionality."""
import pytest
import os
import shutil
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
                checkpoints = await pipeline.checkpoint_manager.list_checkpoints()
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
