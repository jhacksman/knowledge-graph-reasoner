# Knowledge Graph Comparison Report: {{ report_title }}

## Comparison Summary

{{ comparison_summary }}

## Metrics Comparison

### Core Metrics

| Metric | Iteration {{ iter1 }} | Iteration {{ iter2 }} | Change | % Change |
|--------|--------------|--------------|--------|----------|
{% for metric_name, metric_data in comparison.metrics.items() %}
| {{ metric_name | replace("_", " ") | title }} | {{ metric_data.values[0] }} | {{ metric_data.values[1] }} | {{ metric_data.difference }} | {{ metric_data.percent_change }}% |
{% endfor %}

### Domain Coverage

{% if comparison.domain_coverage %}
| Domain | Iteration {{ iter1 }} | Iteration {{ iter2 }} | Change | % Change |
|--------|--------------|--------------|--------|----------|
{% for domain, domain_data in comparison.domain_coverage.items() %}
| {{ domain }} | {{ domain_data.values[0] * 100 }}% | {{ domain_data.values[1] * 100 }}% | {{ domain_data.difference * 100 }}% | {{ domain_data.percent_change }}% |
{% endfor %}
{% else %}
No domain coverage data available for comparison.
{% endif %}

## Visualization

{% if visualization_path %}
![Comparison Visualization]({{ visualization_path }})
{% else %}
No visualization available.
{% endif %}

## Key Insights

{% for metric_name, metric_data in comparison.metrics.items() %}
{% if metric_data.percent_change > 10 %}
- **{{ metric_name | replace("_", " ") | title }}** increased significantly by {{ metric_data.percent_change }}%
{% elif metric_data.percent_change < -10 %}
- **{{ metric_name | replace("_", " ") | title }}** decreased significantly by {{ metric_data.percent_change }}%
{% endif %}
{% endfor %}

{% if comparison.node_overlap %}
### Node Overlap
- Common nodes: {{ comparison.node_overlap.common_count }}
- Nodes only in iteration {{ iter1 }}: {{ comparison.node_overlap.reference_only }}
- Nodes only in iteration {{ iter2 }}: {{ comparison.node_overlap.current_only }}
- Overlap percentage: {{ comparison.node_overlap.overlap_percentage }}%
{% endif %}

{% if comparison.edge_overlap %}
### Edge Overlap
- Common edges: {{ comparison.edge_overlap.common_count }}
- Edges only in iteration {{ iter1 }}: {{ comparison.edge_overlap.reference_only }}
- Edges only in iteration {{ iter2 }}: {{ comparison.edge_overlap.current_only }}
- Overlap percentage: {{ comparison.edge_overlap.overlap_percentage }}%
{% endif %}

---

Generated on {{ timestamp }} by Knowledge Graph Reasoner
