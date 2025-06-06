import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.tree import plot_tree
from typing import List, Optional, Tuple

def plot_formatted_tree(
    decision_tree, 
    feature_names=None,
    class_names=None,
    samples_format="number",    # "percentage" or "number"
    value_format="percentage",  # "percentage" or "number"
    max_decimal_places=1,       # Maximum decimal places for formatting
    integer_thresholds=False,   # Whether to display thresholds as integers
    class_display="all",        # "all" or "one" - how to display class names
    figsize=(20, 10),
    filled=True,                # Whether to fill the nodes with color
    **kwargs  
):
    """
    Plot a decision tree with formatted node information.

    Parameters:
        decision_tree (sklearn.tree.DecisionTreeClassifier): The decision tree to plot.
        feature_names (list): List of feature names to use in the plot.
        class_names (list): List of class names to use in the plot.
        samples_format (str): Format for displaying samples in each node: "percentage" or "number" (default).
        value_format (str): Format for displaying values in each node: "percentage" or "number" (default).
        max_decimal_places (int): Maximum number of decimal places to display in node values (default: 1).
        integer_thresholds (bool): Whether to display thresholds as integers (default: False).
        class_display (str): How to display class names in the plot: "all" or "one" (default).
        figsize (tuple): The size of the figure in inches (default: (20, 10)).
        filled (bool): Whether to fill the nodes with color (default: True).
        **kwargs: Additional arguments to pass to `sklearn.tree.plot_tree()`.
    """
     # Validate input parameters
    if value_format not in ["percentage", "number"]:
        raise ValueError("value_format must be 'percentage' or 'number'")
    if samples_format not in ["percentage", "number"]:
        raise ValueError("samples_format must be 'percentage' or 'number'")
    if class_display not in ["all", "one"]:
        raise ValueError("class_display must be 'all' or 'one'")
        
    # Get total training sample size
    total_samples = int(decision_tree.tree_.n_node_samples[0])
    if total_samples <= 0:
        raise ValueError("Total samples must be greater than 0")
    
    # Create the figure and plot the tree
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=filled, 
        **kwargs
    )
    
    # Find all the text boxes in the tree visualization
    for text in ax.texts:
        content = text.get_text()
        updated_content = content
        
        # Format samples field if present
        if 'samples = ' in content:
            samples_match = re.search(r'samples = (\d+)', content)
            if samples_match:
                node_samples = int(samples_match.group(1))
                
                # Format samples if needed
                if samples_format == "percentage":
                    samples_percent = (node_samples / total_samples) * 100
                    if samples_percent == int(samples_percent):
                        samples_str = f"{int(samples_percent)}%"
                    else:
                        samples_str = f"{samples_percent:.{max_decimal_places}f}%"
                    updated_content = re.sub(
                        r'samples = \d+', 
                        f'samples = {samples_str}', 
                        updated_content
                    )
        
        # Format value field if present
        if 'value = [' in content:
            value_match = re.search(r'value = \[(.*?)\]', updated_content)
            if value_match:
                value_str = value_match.group(1)
                values = [float(v.strip()) for v in value_str.split(',')]
                
                if value_format == "percentage":
                    formatted_values = []
                    for v in values:
                        pct = (v / node_samples) * 100
                        if pct == int(pct):
                            formatted_values.append(f"{int(pct)}%")
                        else:
                            formatted_values.append(f"{pct:.{max_decimal_places}f}%")
                    formatted_values_str = ", ".join(formatted_values)
                    updated_content = re.sub(
                        r'value = \[.*?\]', 
                        f'value = [{formatted_values_str}]',
                        updated_content
                    )
        
        # Format class - handle class display options
        if class_display == "all" and class_names is not None and len(class_names) > 0:
            class_match = re.search(r'class = ([^\n]+)', updated_content)
            if class_match:
                class_str = class_names
                updated_content = re.sub(
                    r'class = ([^\n]+)', 
                    f'class = {class_str}',
                    updated_content
                )
                
        # Format threshold to integer if requested
        if integer_thresholds and ('<=' in content or '>' in content):
            threshold_match = re.search(r'([<=>]+) (\d+\.\d+)', content)
            if threshold_match:
                comparison = threshold_match.group(1)
                threshold = float(threshold_match.group(2))
                
                if comparison == "<=":
                    new_threshold = int(threshold)
                    if new_threshold < threshold:
                        updated_content = updated_content.replace(
                            f"{comparison} {threshold}", 
                            f"<= {new_threshold}"
                        )
                    else:
                        updated_content = updated_content.replace(
                            f"{threshold}", 
                            f"{new_threshold}"
                        )
                elif comparison == ">":
                    new_threshold = int(np.ceil(threshold))
                    if new_threshold > threshold:
                        updated_content = updated_content.replace(
                            f"{comparison} {threshold}", 
                            f"> {new_threshold-1}"
                        )
                    else:
                        updated_content = updated_content.replace(
                            f"{threshold}", 
                            f"{new_threshold}"
                        )
        
        # Update the text if it changed
        if updated_content != content:
            text.set_text(updated_content)
    
    plt.tight_layout()
    return fig, ax


def check_nulls_in_leaf_nodes(df, leaf_node_column, columns_to_check):
    """
    Checks for null values in specific columns for each leaf node and returns a dictionary for the leaf nodes with null values.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        leaf_node_column (str): The column name representing the leaf nodes.
        columns_to_check (list): List of column names to check for null values.

    Returns:
        dict: A dictionary containing information about null values in each leaf node.
        {
            'null_count': int,                # Number of samples with at least one null in the specified columns within this leaf node.
            'sample_indices': list of int,     # List of DataFrame indices for samples with nulls in the specified columns.
            'total_samples_in_leaf': int       # Total number of samples in this leaf node.
        }
        Only leaf nodes containing at least one null value in the specified columns are included in the output dictionary.
    """
    null_by_leaf = {}

    for leaf_node in np.unique(df[leaf_node_column]):
        leaf_samples = df[df[leaf_node_column] == leaf_node]
        null_count = leaf_samples[columns_to_check].isnull().sum().sum()

        if null_count > 0:
            null_indices = leaf_samples[leaf_samples[columns_to_check].isnull().any(axis=1)].index.tolist()
            null_by_leaf[int(leaf_node)] = {
                'null_count': len(null_indices),  # Number of samples with at least one null value in the specified columns
                'sample_indices': null_indices,  # List of DataFrame indices for samples with nulls in the specified columns
                'total_samples_in_leaf': len(leaf_samples)  # Total number of samples in this leaf node
            }

    return null_by_leaf


def get_nulls_in_leaf_nodes(decision_tree, X, df, leaf_column, columns_to_check):
    """
    Analyzes the distribution of null values within specified columns for each leaf node of a trained Decision Tree Model.

    This function assigns each sample to its corresponding leaf node, appends this information to the provided DataFrame, 
    and then inspects the specified columns for null values within each leaf node. It returns a mapping that details, 
    for every leaf node, the number of samples containing at least one null in the specified columns, the indices of these samples, 
    and the total number of samples assigned to that leaf node.

    Parameters:
        decision_tree: Trained Decision Tree Model.
        X: DataFrame or array containing features used for training the decision_tree.
        df: DataFrame to which the leaf node column will be added.
        leaf_column: Name for the new leaf node column in the DataFrame.
        columns_to_check: List of columns to check for null values in each leaf node.

    Returns:
        dict: A mapping from each leaf node to a dictionary containing:
            - 'null_counts': int, number of samples in the leaf node with at least one null in the specified columns.
            - 'sample_indices': list of int, indices of samples with nulls in the specified columns.
            - 'total_samples_in_leaf': int, total number of samples assigned to the leaf node.
    """
    # Get leaf node assignments
    leaf_nodes = decision_tree.apply(X)

    # Add leaf node information to the DataFrame
    df_copy = df.copy()
    df_copy[leaf_column] = leaf_nodes

    # Check for null values in each leaf node
    nulls_in_leaf_nodes = check_nulls_in_leaf_nodes(df_copy, leaf_column, columns_to_check)
    
    return nulls_in_leaf_nodes


def format_threshold(x):
    """
    Format a floating-point number to a string with up to four decimal places, 
    removing any trailing zeros and the decimal point if not needed.

    Parameters:
        x (float): The number to format.

    Returns:
        str: The formatted string representation of the number.
    """
    # Format the number to a string with four decimal places
    s = f"{x:.4f}"
    # Remove trailing zeros and the decimal point if necessary
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def format_threshold(x: float) -> str:
    """
    Format a floating-point number to a string with up to four decimal places,
    removing any trailing zeros and the decimal point if not needed.

    Parameters:
        x (float): The number to format.

    Returns:
        str: The formatted string representation of the number.
    """
    # Format the number to a string with four decimal places
    s = f"{x:.4f}"
    # Remove trailing zeros and the decimal point if necessary
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def summarize_tree(clf,
                   feature_names: Optional[List[str]] = None,
                   class_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize the decision tree into a DataFrame.

    Parameters:
        clf: A trained DecisionTreeClassifier or DecisionTreeRegressor.
        feature_names: List of feature names. If None, use the feature indices.
        class_list: List of class names. If None, use the class indices.

    Returns:
        A DataFrame with the following columns:
            leaf_index: The index of the leaf node.
            feature_name: The name of the feature that splits the node.
            Sample Size: The number of samples in the leaf node.
            class_name: The name of the class. If class_list is None, use class indices.
    """
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value
    n_classes = values.shape[2]

    if class_list is None:
        class_list = [f'Class {i}' for i in range(n_classes)]

    feature_order: List[str] = []
    leaf_data: List[Tuple[int, List[Tuple[int, float, str]], int, List[float]]] = []

    def recurse(node_id: int, path_conditions: List[Tuple[int, float, str]]) -> None:
        """
        Recursively traverse the decision tree and collect information about the leaf nodes.
        """
        if children_left[node_id] == children_right[node_id]:
            # Leaf node
            class_counts = values[node_id][0].tolist()
            total_samples = int(sum(class_counts))
            leaf_data.append((node_id, path_conditions, total_samples, class_counts))
            return
        # Non-leaf node
        feat_id = features[node_id]
        thresh = thresholds[node_id]
        feat_name = feature_names[feat_id] if feature_names else str(feat_id)
        if feat_name not in feature_order:
            feature_order.append(feat_name)
        recurse(children_left[node_id], path_conditions + [(feat_id, thresh, '<=')])
        recurse(children_right[node_id], path_conditions + [(feat_id, thresh, '>')])

    recurse(0, [])

    rows = []
    for leaf_id, conditions, sample_size, class_counts in leaf_data:
        row = {'leaf_index': leaf_id}
        feat_bounds = {}
        for feat_id, thresh, ineq in conditions:
            feat_key = feature_names[feat_id] if feature_names else str(feat_id)
            if feat_key not in feat_bounds:
                feat_bounds[feat_key] = {'lower': None, 'upper': None}
            if ineq == '>':
                if feat_bounds[feat_key]['lower'] is None or thresh > feat_bounds[feat_key]['lower']:
                    feat_bounds[feat_key]['lower'] = thresh
            else:
                if feat_bounds[feat_key]['upper'] is None or thresh < feat_bounds[feat_key]['upper']:
                    feat_bounds[feat_key]['upper'] = thresh
        for feat in feature_order:
            if feat in feat_bounds:
                b = feat_bounds[feat]
                parts = []
                if b['lower'] is not None:
                    parts.append(f"> {format_threshold(b['lower'])}")
                if b['upper'] is not None:
                    parts.append(f"<= {format_threshold(b['upper'])}")
                row[feat] = ', '.join(parts)
            else:
                row[feat] = ''

        row['Sample Size'] = sample_size
        for i, cls in enumerate(class_list):
            row[cls] = int(class_counts[i])

        rows.append(row)

    col_order = ['leaf_index'] + feature_order + ['Sample Size'] + class_list
    return pd.DataFrame(rows)[col_order]

