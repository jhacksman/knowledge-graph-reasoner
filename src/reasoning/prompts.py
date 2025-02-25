"""Structured prompts for knowledge graph reasoning."""

CONCEPT_GENERATION_PROMPT = """Given the current knowledge graph state:
- Modularity: {modularity}
- Average Path Length: {avg_path_length}
- Bridge Nodes: {bridge_nodes}

Generate new concepts and relationships related to: {seed_concept}

Focus on concepts that would:
1. Strengthen existing knowledge clusters
2. Create meaningful bridges between domains
3. Maintain network stability

For each concept, provide:
- A clear name
- A detailed description
- Relationships to other concepts (both existing and new)

Format your response with entities and relationships in this format:
<entity>entity_name: entity_description</entity>
<relationship>source_entity: target_entity: relationship_type: description</relationship>

Example:
<entity>Neural Networks: Computational models inspired by the human brain's structure</entity>
<entity>Deep Learning: A subset of machine learning using neural networks with many layers</entity>
<relationship>Neural Networks: Deep Learning: is_parent_of: Neural networks are the foundational concept that deep learning builds upon</relationship>"""

ITERATIVE_REASONING_PROMPT = """You are expanding a knowledge graph through iterative reasoning.

Current focus concept: {focus_concept}

Graph context:
- Related concepts: {related_concepts}
- Bridge nodes: {bridge_nodes}
- Current modularity: {modularity}
- Average path length: {avg_path_length}

Your task is to:
1. Generate new concepts that extend or connect to the focus concept
2. Create meaningful relationships between concepts
3. Consider both depth (specialization) and breadth (interdisciplinary connections)

Format your response with entities and relationships in this format:
<entity>entity_name: entity_description</entity>
<relationship>source_entity: target_entity: relationship_type: description</relationship>"""

COMPOSITIONAL_REASONING_PROMPT = """You are performing compositional reasoning over a knowledge graph.

Given these concepts:
{concepts}

And these relationships:
{relationships}

Your task is to:
1. Identify higher-level patterns or abstractions that emerge from these concepts
2. Synthesize new composite concepts that integrate multiple existing ideas
3. Discover potential synergies or emergent properties

Format your response with entities and relationships in this format:
<entity>entity_name: entity_description</entity>
<relationship>source_entity: target_entity: relationship_type: description</relationship>"""

RELATIONSHIP_INFERENCE_PROMPT = """Analyze the relationship between these concepts:
Concept A: {concept_a}
Concept B: {concept_b}

Current graph context:
{graph_context}

Determine:
1. If a relationship exists
2. The type of relationship (e.g., is_a, part_of, causes, enables, contradicts)
3. The strength/confidence of this relationship
4. A detailed description of how these concepts relate

Format your response as:
<relationship>concept_a: concept_b: relationship_type: detailed description of the relationship</relationship>"""

BRIDGE_NODE_PROMPT = """Analyze potential bridge concepts between domains:
Domain A: {domain_a}
Domain B: {domain_b}

Current bridge nodes: {bridge_nodes}
Modularity score: {modularity}

Generate concepts that could:
1. Connect these domains meaningfully
2. Maintain graph stability
3. Enhance knowledge transfer

Format your response with entities and relationships in this format:
<entity>bridge_concept: detailed description of how this concept bridges the domains</entity>
<relationship>bridge_concept: domain_a_concept: relationship_type: description</relationship>
<relationship>bridge_concept: domain_b_concept: relationship_type: description</relationship>"""

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
