# format_tree

A Python package for advanced formatting, analysis, and visualization of scikit-learn decision trees.

## Features
- Plot decision trees with customizable node information (samples, class values as number or percentage)
- Option to display thresholds as integers
- Flexible class name display
- Analyze and summarize tree structure and leaf nodes
- Utilities for checking and reporting null values in leaf nodes
- Convert tree structure to DataFrame for further analysis

## Installation
```bash
pip install format_tree
```

## Usage
### Plot a formatted decision tree
```python
from format_tree import plot_formatted_tree
fig, ax = plot_formatted_tree(
    decision_tree, 
    feature_names=feature_names, 
    class_names=class_names,
    samples_format="percentage",
    value_format="number",
    integer_thresholds=True
)
fig.show()
```

### Check for nulls in leaf nodes
```python
from format_tree import get_nulls_in_leaf_nodes
nulls = get_nulls_in_leaf_nodes(
    decision_tree, X, df, leaf_column="leaf_node", columns_to_check=["feature1", "feature2"]
)
print(nulls)
```

### Summarize tree structure as a DataFrame
```python
from format_tree import summarize_tree
summary_df = summarize_tree(decision_tree, feature_names=feature_names, class_list=class_names)
print(summary_df.head())
```

## License
MIT

## PyPI
https://pypi.org/project/format_tree/

## GitHub for more information and examples
https://github.com/k-explore/format_tree
