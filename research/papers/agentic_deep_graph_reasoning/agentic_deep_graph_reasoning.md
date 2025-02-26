# Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks

**Markus J. Buehler**  
Laboratory for Atomistic and Molecular Mechanics  
Center for Computational Science and Engineering  
Schwarzman College of Computing  
Massachusetts Institute of Technology  
Cambridge, MA 02139, USA  
mbuehler@MIT.EDU  

*Corresponding author.*

## Abstract

We present an agentic, autonomous graph expansion framework that iteratively structures and refines knowledge *in situ*. Unlike conventional knowledge graph construction methods relying on static extraction or single-pass learning, our approach couples a reasoning-native large language model with a continually updated graph representation. At each step, the system actively generates new concepts and relationships, merges them into a global graph, and formulates subsequent prompts based on its evolving structure. Through this feedback-driven loop, the model organizes information into a scale-free network characterized by hub formation, stable modularity, and bridging nodes that link disparate knowledge clusters. Over hundreds of iterations, new nodes and edges continue to appear without saturating, while centrality measures and shortest path distributions evolve to yield increasingly distributed connectivity. Our analysis reveals emergent patterns—such as the rise of highly connected "hub" concepts and the shifting influence of "bridge" nodes—indicating that agentic, self-reinforcing graph construction can yield open-ended, coherent knowledge structures. Applied to materials design problems, we present compositional reasoning experiments by extracting node-specific and synergy-level principles to foster genuinely novel knowledge synthesis, yielding cross-domain ideas that transcend rote summarization and strengthen the framework's potential for open-ended scientific discovery. We discuss other applications in scientific discovery and outline future directions for enhancing scalability and interpretability.

*Keywords*: Artificial Intelligence · Science · Graph Theory · Category Theory · Materials Science · Materiomics · Language Modeling · Reasoning · Isomorphisms · Engineering

## 1 Introduction

Scientific inquiry often proceeds through an interplay of incremental refinement and transformative leaps, evoking broader questions of how knowledge evolves under continual reflection and questioning. In many accounts of discovery, sustained progress arises not from isolated insights but from an iterative process in which prior conclusions are revisited, expressed as generalizable ideas, refined, or even reorganized as new evidence and perspectives emerge. Foundational work in category theory has formalized aspects of this recursive structuring, showing how hierarchical representations can unify diverse knowledge domains and enable higher-level abstractions in both the natural and social sciences. Across engineering disciplines including materials science, such iterative integration of information has proven essential in synthesizing deeply interlinked concepts.

Recent AI methods, however, often emphasize predictive accuracy and single-step outputs over the layered, self-reflective processes that characterize human problem-solving. Impressive gains in natural language processing, multimodal reasoning, and materials science, including breakthroughs in molecular biology and protein folding, showcase the prowess of large-scale models trained on vast datasets. Yet most of the early systems generate answers in a single pass, sidestepping the symbolic, stepwise reasoning that often underpins scientific exploration. This gap has prompted a line of research into modeling that explicitly incorporates relational modeling, reflection or multi-step inferences, hinting at a transition from single-shot pattern recognition to more adaptive synthesis of answers from first principles in ways that more closely resemble compositional mechanisms.

## 2 Results and Discussion

We present the results of experiments in which the graph-native reasoning model engages in a continuous, recursive process of graph-based reasoning, expanding its knowledge graph representation autonomously over 1,000 iterations. Unlike prior approaches that rely on a small number of just a few recursive reasoning steps, the experiments reported in this paper explore how knowledge formation unfolds in an open-ended manner, generating a dynamically evolving graph. As the system iterates, it formulates new tasks, refines reasoning pathways, and integrates emerging concepts, progressively structuring its own knowledge representation.

The recursive graph reasoning process can be conducted in either an open-ended setting or developed into a more tailored manner to address a specific domain or flavor in which reasoning steps are carried out. In the example explored here, we focus on designing impact-resistant materials. In this specialized scenario, we initiate the model with a concise, topic-specific prompt – e.g., Describe a way to design impact resistant materials, and maintain the iterative process of extracting structured knowledge from the model's reasoning.

## Code, data and model weights availability

Codes, model weights and additional materials are available at https://huggingface.co/lamm-mit and https://github.com/lamm-mit/PRefLexOR. The model used for the experiments is available at lamm-mit/Graph-Preflexor_01062025.

## Conflicts of Interest

The author declares no conflicts of interest of any kind.

## Acknowledgments

The author acknowledges support from the MIT Generative AI initiative.

*Note: This is a condensed version of the original paper. The full paper contains additional sections on methodology, detailed results, and extensive references.*
