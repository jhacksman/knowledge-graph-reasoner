"""Custom exceptions for vector store."""

class MilvusError(Exception):
    """Base exception for Milvus operations."""
    pass


class CollectionInitError(MilvusError):
    """Error initializing Milvus collection."""
    pass


class SearchError(MilvusError):
    """Error during vector search."""
    pass
