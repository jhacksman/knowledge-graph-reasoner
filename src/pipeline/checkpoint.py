"""Checkpoint manager for saving and resuming graph expansion sessions."""
from typing import Dict, Any, Optional, List, Union, Tuple
import os
import json
import time
import asyncio
import logging
import hashlib
import datetime
from pathlib import Path
import pickle
import gzip
import shutil

from ..graph.manager import GraphManager
from ..vector_store.base import BaseVectorStore
from ..metrics.graph_metrics import GraphMetrics

logger = logging.getLogger(__name__)


class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    def __init__(
        self,
        timestamp: float,
        iteration: int,
        config: Dict[str, Any],
        version: str = "1.0.0",
        description: str = "",
        checksum: Optional[str] = None
    ):
        """Initialize checkpoint metadata.
        
        Args:
            timestamp: Unix timestamp when checkpoint was created
            iteration: Iteration number of the reasoning process
            config: Configuration used for the reasoning process
            version: Checkpoint format version
            description: Optional description of the checkpoint
            checksum: Optional checksum of the checkpoint data
        """
        self.timestamp = timestamp
        self.iteration = iteration
        self.config = config
        self.version = version
        self.description = description
        self.checksum = checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.
        
        Returns:
            Dict[str, Any]: Metadata as dictionary
        """
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.datetime.fromtimestamp(self.timestamp).isoformat(),
            "iteration": self.iteration,
            "config": self.config,
            "version": self.version,
            "description": self.description,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create metadata from dictionary.
        
        Args:
            data: Dictionary with metadata
            
        Returns:
            CheckpointMetadata: Metadata object
        """
        return cls(
            timestamp=data["timestamp"],
            iteration=data["iteration"],
            config=data["config"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            checksum=data.get("checksum")
        )


class CheckpointManager:
    """Manager for saving and resuming graph expansion sessions."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        graph_manager: GraphManager,
        config: Dict[str, Any],
        checkpoint_interval_iterations: int = 10,
        checkpoint_interval_minutes: float = 30.0,
        max_checkpoints: int = 5,
        compress: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for storing checkpoints
            graph_manager: Graph manager to checkpoint
            config: Configuration used for the reasoning process
            checkpoint_interval_iterations: Checkpoint every N iterations
            checkpoint_interval_minutes: Checkpoint every N minutes
            max_checkpoints: Maximum number of checkpoints to keep
            compress: Whether to compress checkpoint files
        """
        self.base_dir = Path(base_dir)
        self.graph_manager = graph_manager
        self.config = config
        self.checkpoint_interval_iterations = checkpoint_interval_iterations
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.max_checkpoints = max_checkpoints
        self.compress = compress
        
        # Create checkpoint directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last checkpoint time
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_iteration = 0
    
    def get_checkpoint_path(self, iteration: int) -> Path:
        """Get path for checkpoint directory.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Path: Path to checkpoint directory
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.base_dir / f"checkpoint_{timestamp}_{iteration:04d}"
    
    async def should_checkpoint(self, iteration: int) -> bool:
        """Check if checkpoint should be created.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            bool: Whether checkpoint should be created
        """
        # Check if enough iterations have passed
        iterations_passed = iteration - self.last_checkpoint_iteration
        if iterations_passed >= self.checkpoint_interval_iterations:
            return True
        
        # Check if enough time has passed
        time_passed = time.time() - self.last_checkpoint_time
        if time_passed >= self.checkpoint_interval_minutes * 60:
            return True
        
        return False
    
    async def create_checkpoint(self, iteration: int, description: str = "") -> Path:
        """Create checkpoint of current graph state.
        
        Args:
            iteration: Current iteration number
            description: Optional description of the checkpoint
            
        Returns:
            Path: Path to created checkpoint
        """
        checkpoint_path = self.get_checkpoint_path(iteration)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating checkpoint at {checkpoint_path}")
        
        try:
            # Save graph structure
            await self._save_graph_structure(checkpoint_path)
            
            # Save embeddings
            await self._save_embeddings(checkpoint_path)
            
            # Save metrics history
            await self._save_metrics(checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                timestamp=time.time(),
                iteration=iteration,
                config=self.config,
                description=description
            )
            
            # Calculate checksum
            checksum = await self._calculate_checksum(checkpoint_path)
            metadata.checksum = checksum
            
            # Save metadata
            await self._save_metadata(checkpoint_path, metadata)
            
            # Update last checkpoint time and iteration
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_iteration = iteration
            
            # Clean up old checkpoints
            await self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint created successfully at {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            # Clean up failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    async def _save_graph_structure(self, checkpoint_path: Path) -> None:
        """Save graph structure to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Get all nodes and edges from the vector store
        # This is a simplified implementation - in a real system,
        # you would need to implement proper serialization for the graph structure
        
        # For now, we'll just save a placeholder
        graph_path = checkpoint_path / "graph_structure.json"
        with open(graph_path, "w") as f:
            json.dump({"nodes": [], "edges": []}, f)
    
    async def _save_embeddings(self, checkpoint_path: Path) -> None:
        """Save embeddings to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Save embeddings in an efficient binary format
        # This is a simplified implementation - in a real system,
        # you would need to implement proper serialization for the embeddings
        
        # For now, we'll just save a placeholder
        embeddings_path = checkpoint_path / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump({"embeddings": []}, f)
        
        # Compress if enabled
        if self.compress:
            with open(embeddings_path, "rb") as f_in:
                with gzip.open(f"{embeddings_path}.gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove uncompressed file
            os.remove(embeddings_path)
    
    async def _save_metrics(self, checkpoint_path: Path) -> None:
        """Save metrics history to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Save metrics history
        # This is a simplified implementation - in a real system,
        # you would need to implement proper serialization for the metrics history
        
        # For now, we'll just save a placeholder
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"metrics_history": []}, f)
    
    async def _save_metadata(self, checkpoint_path: Path, metadata: CheckpointMetadata) -> None:
        """Save metadata to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            metadata: Checkpoint metadata
        """
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    async def _calculate_checksum(self, checkpoint_path: Path) -> str:
        """Calculate checksum for checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            str: Checksum
        """
        checksums = []
        
        # Calculate checksums for all files in checkpoint directory
        for path in checkpoint_path.glob("**/*"):
            if path.is_file() and path.name != "metadata.json":
                with open(path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums.append(f"{path.name}:{file_hash}")
        
        # Sort checksums for deterministic result
        checksums.sort()
        
        # Calculate overall checksum
        return hashlib.sha256("|".join(checksums).encode()).hexdigest()
    
    async def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        # Get all checkpoint directories
        checkpoints = []
        for path in self.base_dir.glob("checkpoint_*"):
            if path.is_dir():
                # Try to parse metadata
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        checkpoints.append((path, metadata["timestamp"]))
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata for checkpoint {path}: {e}")
        
        # Sort checkpoints by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for path, _ in checkpoints[self.max_checkpoints:]:
            logger.info(f"Removing old checkpoint: {path}")
            shutil.rmtree(path)
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint metadata
        """
        checkpoints = []
        
        # Get all checkpoint directories
        for path in self.base_dir.glob("checkpoint_*"):
            if path.is_dir():
                # Try to parse metadata
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        checkpoints.append({
                            "path": str(path),
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata for checkpoint {path}: {e}")
        
        # Sort checkpoints by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
        
        return checkpoints
    
    async def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[bool, str]:
        """Load checkpoint and restore graph state.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple[bool, str]: Success flag and message
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False, f"Checkpoint not found: {checkpoint_path}"
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Verify checkpoint integrity
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                return False, f"Metadata not found in checkpoint: {checkpoint_path}"
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            
            # Verify checksum if available
            if metadata.checksum:
                calculated_checksum = await self._calculate_checksum(checkpoint_path)
                if calculated_checksum != metadata.checksum:
                    return False, f"Checkpoint integrity check failed: {checkpoint_path}"
            
            # Load graph structure
            # This is a simplified implementation - in a real system,
            # you would need to implement proper deserialization for the graph structure
            
            # Load embeddings
            # This is a simplified implementation - in a real system,
            # you would need to implement proper deserialization for the embeddings
            
            # Load metrics history
            # This is a simplified implementation - in a real system,
            # you would need to implement proper deserialization for the metrics history
            
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            return True, f"Checkpoint loaded successfully from {checkpoint_path}"
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False, f"Failed to load checkpoint: {e}"
    
    async def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple[bool, str]: Validation result and message
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False, f"Checkpoint not found: {checkpoint_path}"
        
        try:
            # Check if metadata exists
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                return False, f"Metadata not found in checkpoint: {checkpoint_path}"
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            
            # Verify checksum if available
            if metadata.checksum:
                calculated_checksum = await self._calculate_checksum(checkpoint_path)
                if calculated_checksum != metadata.checksum:
                    return False, f"Checkpoint integrity check failed: {checkpoint_path}"
            
            # Check if required files exist
            required_files = [
                "graph_structure.json",
                "metrics.json"
            ]
            
            # Check for embeddings file (compressed or uncompressed)
            if not (checkpoint_path / "embeddings.pkl").exists() and not (checkpoint_path / "embeddings.pkl.gz").exists():
                return False, f"Embeddings file not found in checkpoint: {checkpoint_path}"
            
            for file in required_files:
                if not (checkpoint_path / file).exists():
                    return False, f"Required file not found in checkpoint: {file}"
            
            return True, f"Checkpoint validation successful: {checkpoint_path}"
            
        except Exception as e:
            logger.error(f"Failed to validate checkpoint: {e}")
            return False, f"Failed to validate checkpoint: {e}"
