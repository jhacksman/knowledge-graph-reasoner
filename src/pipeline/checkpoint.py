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
        checksum: Optional[str] = None,
        incremental: bool = False
    ):
        """Initialize checkpoint metadata.
        
        Args:
            timestamp: Unix timestamp when checkpoint was created
            iteration: Iteration number of the reasoning process
            config: Configuration used for the reasoning process
            version: Checkpoint format version
            description: Optional description of the checkpoint
            checksum: Optional checksum of the checkpoint data
            incremental: Whether this is an incremental checkpoint
        """
        self.timestamp = timestamp
        self.iteration = iteration
        self.config = config
        self.version = version
        self.description = description
        self.checksum = checksum
        self.incremental = incremental
    
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
            "checksum": self.checksum,
            "incremental": self.incremental
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
            checksum=data.get("checksum"),
            incremental=data.get("incremental", False)
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
            
            # Compress graph structure if enabled
            if self.compress:
                structure_path = checkpoint_path / "graph_structure.json"
                compressed_path = checkpoint_path / "graph_structure.json.gz"
                if structure_path.exists():
                    with open(structure_path, "rb") as f_in:
                        with gzip.open(compressed_path, "wb") as f_out:
                            f_out.write(f_in.read())
                    structure_path.unlink()  # Remove uncompressed file
            
            # Save embeddings
            await self._save_embeddings(checkpoint_path)
            
            # Save metrics history
            await self._save_metrics(checkpoint_path)
            
            # Compress metrics if enabled
            if self.compress:
                metrics_path = checkpoint_path / "metrics.json"
                compressed_path = checkpoint_path / "metrics.json.gz"
                if metrics_path.exists():
                    with open(metrics_path, "rb") as f_in:
                        with gzip.open(compressed_path, "wb") as f_out:
                            f_out.write(f_in.read())
                    metrics_path.unlink()  # Remove uncompressed file
            
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
    
    async def _save_graph_structure(self, checkpoint_path: Path, chunk_size: int = 1000) -> None:
        """Save graph structure to checkpoint with streaming support.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            chunk_size: Number of nodes/edges to process at once
        """
        structure_path = checkpoint_path / "graph_structure.json"
        
        # Check if streaming methods are available
        if hasattr(self.graph_manager, 'get_nodes_stream') and hasattr(self.graph_manager, 'get_edges_stream'):
            # Use streaming to handle large graphs
            with open(structure_path, "w") as f:
                f.write('{"nodes": [')
                
                # Write nodes in chunks
                node_count = 0
                async for nodes_chunk in self.graph_manager.get_nodes_stream(chunk_size):
                    if node_count > 0:
                        f.write(',')
                    f.write(','.join([json.dumps(node.to_dict()) for node in nodes_chunk]))
                    node_count += len(nodes_chunk)
                
                f.write('], "edges": [')
                
                # Write edges in chunks
                edge_count = 0
                async for edges_chunk in self.graph_manager.get_edges_stream(chunk_size):
                    if edge_count > 0:
                        f.write(',')
                    f.write(','.join([json.dumps(edge.to_dict()) for edge in edges_chunk]))
                    edge_count += len(edges_chunk)
                
                f.write(']}')
        else:
            # Fall back to standard export if streaming methods are not available
            await self.graph_manager.export_graph_structure(structure_path, compress=False)
    
    async def _save_embeddings(self, checkpoint_path: Path) -> None:
        """Save embeddings to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Export embeddings using GraphManager
        embeddings_path = checkpoint_path / "embeddings.pkl"
        await self.graph_manager.export_embeddings(embeddings_path, compress=self.compress)
    
    async def _save_metrics(self, checkpoint_path: Path) -> None:
        """Save metrics history to checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Export metrics history using GraphManager
        metrics_path = checkpoint_path / "metrics.json"
        await self.graph_manager.export_metrics_history(metrics_path)
    
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
    
    async def _validate_graph_consistency(self, checkpoint_path: Path) -> Tuple[bool, str]:
        """Validate graph consistency (no dangling edges, etc.).
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple[bool, str]: Validation result and message
        """
        try:
            # Load graph structure
            graph_path = checkpoint_path / "graph_structure.json"
            if not graph_path.exists():
                return False, f"Graph structure file not found: {graph_path}"
            
            with open(graph_path, "r") as f:
                graph_data = json.load(f)
            
            # Check if nodes and edges are present
            if "nodes" not in graph_data or "edges" not in graph_data:
                return False, "Invalid graph structure format: missing nodes or edges"
            
            # Extract node IDs and edge references
            node_ids = set()
            for node in graph_data["nodes"]:
                if "id" not in node:
                    return False, "Invalid node format: missing ID"
                node_ids.add(node["id"])
            
            # Check for dangling edges
            for edge in graph_data["edges"]:
                if "source" not in edge or "target" not in edge:
                    return False, "Invalid edge format: missing source or target"
                
                if edge["source"] not in node_ids:
                    return False, f"Dangling edge: source node {edge['source']} not found"
                
                if edge["target"] not in node_ids:
                    return False, f"Dangling edge: target node {edge['target']} not found"
            
            return True, "Graph consistency validation successful"
            
        except Exception as e:
            logger.error(f"Failed to validate graph consistency: {e}")
            return False, f"Failed to validate graph consistency: {e}"
    
    async def _handle_version_compatibility(self, checkpoint_path: Path, version: str) -> None:
        """Handle version compatibility for checkpoint format changes.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            version: Checkpoint format version
        """
        current_version = "1.0.0"
        
        # Parse versions
        checkpoint_version = tuple(map(int, version.split(".")))
        current_version_tuple = tuple(map(int, current_version.split(".")))
        
        # Handle version-specific transformations
        if checkpoint_version < current_version_tuple:
            logger.info(f"Upgrading checkpoint from version {version} to {current_version}")
            
            # Example: If we're upgrading from 0.x.x to 1.0.0
            if checkpoint_version[0] == 0 and current_version_tuple[0] == 1:
                await self._upgrade_from_v0_to_v1(checkpoint_path)
        elif checkpoint_version > current_version_tuple:
            logger.warning(f"Downgrading checkpoint from version {version} to {current_version}")
            
            # Example: If we're downgrading from 2.x.x to 1.0.0
            if checkpoint_version[0] == 2 and current_version_tuple[0] == 1:
                await self._downgrade_from_v2_to_v1(checkpoint_path)
    
    async def _upgrade_from_v0_to_v1(self, checkpoint_path: Path) -> None:
        """Upgrade checkpoint from version 0.x.x to 1.0.0.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # This is a placeholder for version-specific upgrade logic
        # In a real implementation, you would transform the checkpoint data
        # to match the new format
        logger.info(f"Upgrading checkpoint at {checkpoint_path} from v0.x.x to v1.0.0")
        
        # Example: Update graph structure format
        graph_path = checkpoint_path / "graph_structure.json"
        if graph_path.exists():
            with open(graph_path, "r") as f:
                graph_data = json.load(f)
            
            # Apply transformations to match v1.0.0 format
            # For example, rename fields, add new required fields, etc.
            
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)
    
    async def _downgrade_from_v2_to_v1(self, checkpoint_path: Path) -> None:
        """Downgrade checkpoint from version 2.x.x to 1.0.0.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # This is a placeholder for version-specific downgrade logic
        # In a real implementation, you would transform the checkpoint data
        # to match the older format
        logger.info(f"Downgrading checkpoint at {checkpoint_path} from v2.x.x to v1.0.0")
        
        # Example: Update graph structure format
        graph_path = checkpoint_path / "graph_structure.json"
        if graph_path.exists():
            with open(graph_path, "r") as f:
                graph_data = json.load(f)
            
            # Apply transformations to match v1.0.0 format
            # For example, remove fields that don't exist in v1.0.0, etc.
            
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)
    
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
    
    async def create_incremental_checkpoint(self, iteration: int, description: str = "") -> Path:
        """Create incremental checkpoint of current graph state.
        
        Args:
            iteration: Current iteration number
            description: Optional description of the checkpoint
            
        Returns:
            Path: Path to created checkpoint
        """
        checkpoint_path = self.get_checkpoint_path(iteration)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating incremental checkpoint at {checkpoint_path}")
        
        try:
            # Get changes since last checkpoint
            last_checkpoint = None
            checkpoints = await self.list_checkpoints()
            if checkpoints:
                last_checkpoint = Path(checkpoints[0]["path"])
            
            # Save only changed nodes and edges
            if last_checkpoint:
                changes = await self.graph_manager.get_changes_since(
                    self.last_checkpoint_time
                )
                
                # Save changes
                changes_path = checkpoint_path / "changes.json"
                with open(changes_path, "w") as f:
                    json.dump(changes, f)
                
                # Save reference to base checkpoint
                base_path = checkpoint_path / "base_checkpoint.txt"
                with open(base_path, "w") as f:
                    f.write(str(last_checkpoint))
            else:
                # No previous checkpoint, create full checkpoint
                await self._save_graph_structure(checkpoint_path)
                await self._save_embeddings(checkpoint_path)
            
            # Save metrics history
            await self._save_metrics(checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                timestamp=time.time(),
                iteration=iteration,
                config=self.config,
                description=description,
                incremental=last_checkpoint is not None
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
            
            logger.info(f"Incremental checkpoint created successfully at {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create incremental checkpoint: {e}")
            # Clean up failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
            
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
            # Validate checkpoint first
            valid, message = await self.validate_checkpoint(checkpoint_path)
            if not valid:
                return False, message
            
            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            
            # Handle version compatibility
            if metadata.version != "1.0.0":
                # Apply version-specific transformations if needed
                await self._handle_version_compatibility(checkpoint_path, metadata.version)
            
            # Check if this is an incremental checkpoint
            if metadata.incremental:
                # Load base checkpoint first
                base_path_file = checkpoint_path / "base_checkpoint.txt"
                if base_path_file.exists():
                    with open(base_path_file, "r") as f:
                        base_path = Path(f.read().strip())
                    
                    # Load base checkpoint
                    base_success, base_message = await self.load_checkpoint(base_path)
                    if not base_success:
                        logger.warning(f"Failed to load base checkpoint: {base_message}")
                    
                    # Apply changes
                    changes_path = checkpoint_path / "changes.json"
                    if changes_path.exists():
                        with open(changes_path, "r") as f:
                            changes = json.load(f)
                        
                        # Apply changes to graph
                        await self.graph_manager.apply_changes(changes)
                    else:
                        return False, "Changes file not found in incremental checkpoint"
                else:
                    return False, "Base checkpoint reference not found in incremental checkpoint"
            else:
                # Import full graph from checkpoint
                import_result = await self.graph_manager.import_full_graph(
                    directory=checkpoint_path,
                    import_structure=True,
                    import_embeddings=True,
                    import_metrics=True
                )
            
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
            
            # Version compatibility check
            current_version = "1.0.0"  # Current checkpoint format version
            if metadata.version != current_version:
                # Handle version differences
                logger.warning(f"Checkpoint version mismatch: {metadata.version} vs {current_version}")
                # For now, we'll just warn but continue - in a real implementation,
                # you would implement version-specific loading logic here
                
                # Check if the version is compatible (major version must match)
                checkpoint_major = metadata.version.split(".")[0]
                current_major = current_version.split(".")[0]
                if checkpoint_major != current_major:
                    return False, f"Incompatible checkpoint version: {metadata.version} (current: {current_version})"
            
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
            
            # Validate graph consistency (no dangling edges)
            graph_consistency_result = await self._validate_graph_consistency(checkpoint_path)
            if not graph_consistency_result[0]:
                return graph_consistency_result
            
            return True, f"Checkpoint validation successful: {checkpoint_path}"
            
        except Exception as e:
            logger.error(f"Failed to validate checkpoint: {e}")
            return False, f"Failed to validate checkpoint: {e}"
