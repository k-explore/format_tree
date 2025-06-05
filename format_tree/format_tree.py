import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.tree import plot_tree

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
