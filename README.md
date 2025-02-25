# Knowledge Graph Reasoner

An implementation of agentic deep graph reasoning for self-organizing knowledge networks, based on the research paper ["Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks"](https://arxiv.org/pdf/2502.13025v1).

## Overview

This project implements an autonomous graph expansion framework that demonstrates self-organizing intelligence-like behavior through iterative reasoning. The system uses the Venice.ai API with the deepseek-r1-671b model to generate and maintain dynamic knowledge networks without predefined ontologies or centralized control.

## Key Features

- **Autonomous Knowledge Graph Construction**
  - Dynamic concept generation and relationship discovery
  - Self-organizing network structure
  - Interdisciplinary connection formation
  - Bridge node identification and tracking

- **Iterative Reasoning Process**
  - Recursive graph expansion
  - Structured knowledge integration
  - Automated relationship discovery
  - Dynamic prompt generation and refinement

- **Advanced Analytics**
  - Network metric monitoring
  - Community structure analysis
  - Path length optimization
  - Bridge node persistence tracking

## Technical Architecture

### Core Components

1. **Venice.ai Integration Layer**
   - API client for deepseek-r1-671b model
   - Structured prompt management
   - Response parsing and extraction

2. **Graph Database**
   - NetworkX-based implementation
   - Dynamic node and edge management
   - Metadata and attribute support
   - Efficient querying capabilities

3. **Knowledge Extraction Pipeline**
   - Entity-relationship extraction
   - Structured knowledge parsing
   - Graph update management
   - Deduplication handling

4. **Analysis Module**
   - Centrality computations
   - Community detection
   - Path analysis
   - Scale-free properties validation

## Project Structure

```
knowledge-graph-reasoner/
├── research/               # Research materials and documentation
│   ├── papers/            # Academic papers and references
│   └── notes/             # Implementation notes and analysis
├── src/                   # Source code
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Getting Started

### Prerequisites

- Python 3.12+
- Venice.ai API access
- NetworkX
- FastAPI (for API endpoints)

### Installation

#### Using uv (Recommended for macOS)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/jhacksman/knowledge-graph-reasoner.git
cd knowledge-graph-reasoner

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Venice.ai API credentials
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/jhacksman/knowledge-graph-reasoner.git
cd knowledge-graph-reasoner

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Venice.ai API credentials
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{buehler2025agentic,
  title={Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks},
  author={Buehler, M. J.},
  journal={arXiv preprint arXiv:2502.13025},
  year={2025}
}
```

## Acknowledgments

- Based on research by M. J. Buehler
- Uses the Venice.ai API with deepseek-r1-671b model
- Built with NetworkX for graph operations
