# format_tree

A Python package for advanced formatting, analysis, and visualization of scikit-learn decision trees.

## Features
- Plot decision trees with customizable node information (samples, class values as number or percentage)
- Option to display thresholds as integers
- Flexible class name display
- Analyze and summarize tree structure and leaf nodes
- Utilities for checking and reporting null values in leaf nodes (per leaf, per column)
- Convert tree structure to DataFrame for further analysis, including path and sample details
- Support for missing value analysis and advanced DataFrame summaries

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
    integer_thresholds=True,
    display_missing=True,  # Show missing value info if desired
    X_train=X,            # Pass your feature data if using display_missing
    columns_to_check=["feature1", "feature2"]  # Columns to check for nulls
)
fig.show()
```

### Check for nulls in leaf nodes (per leaf, per column)
```python
from format_tree import get_nulls_in_leaf_nodes
nulls = get_nulls_in_leaf_nodes(
    decision_tree, X, ["feature1", "feature2"], df=df  # df is optional, X is used if not provided
)
print(nulls)
```

### Summarize tree structure as a DataFrame
```python
from format_tree import summarize_tree
summary_df = summarize_tree(
    decision_tree,
    feature_names=feature_names,
    class_list=class_names,
    display_missing=True,      # Show missing value info if desired
    X_train=X,                 # Pass your tarining data if using display_missing
    columns_to_check=["feature1", "feature2"]
)
print(summary_df.head())
```

## License
MIT

## PyPI
https://pypi.org/project/format_tree/

## GitHub
https://github.com/k_explore/format_tree
