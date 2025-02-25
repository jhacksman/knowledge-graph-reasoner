"""Structured prompts for knowledge graph reasoning."""

CONCEPT_GENERATION_PROMPT = """Given the current knowledge graph state:
- Modularity: {modularity}
- Average Path Length: {avg_path_length}
- Bridge Nodes: {bridge_nodes}

Generate new concepts related to: {seed_concept}

Focus on concepts that would:
1. Strengthen existing knowledge clusters
2. Create meaningful bridges between domains
3. Maintain network stability

Format: Return a list of concepts, one per line."""

RELATIONSHIP_INFERENCE_PROMPT = """Analyze the relationship between these concepts:
Concept A: {concept_a}
Concept B: {concept_b}

Current graph context:
{graph_context}

Determine:
1. If a relationship exists
2. The type of relationship
3. The strength/confidence of this relationship

Format: Return as JSON with fields: exists (bool), type (str), confidence (float)"""

BRIDGE_NODE_PROMPT = """Analyze potential bridge concepts between domains:
Domain A: {domain_a}
Domain B: {domain_b}

Current bridge nodes: {bridge_nodes}
Modularity score: {modularity}

Generate concepts that could:
1. Connect these domains meaningfully
2. Maintain graph stability
3. Enhance knowledge transfer

Format: Return a list of bridge concepts with explanations, one per line:
concept: explanation"""

STABILITY_CHECK_PROMPT = """Analyze graph stability metrics:
Current modularity: {modularity}
Average path length: {avg_path_length}
Graph diameter: {diameter}
Bridge node count: {bridge_node_count}

Target ranges:
Path length: 4.5-5.0
Diameter: 16-18

Assess:
1. Current stability state
2. Areas needing improvement
3. Recommended actions

Format: Return as JSON with fields:
{{
    "stable": bool,
    "metrics_in_range": List[str],
    "improvements_needed": List[str],
    "recommendations": List[str]
}}"""
