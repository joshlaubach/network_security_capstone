"""
utils.py
--------
General-purpose utility functions for the Data Science Capstone project.

Includes:
- Logging
- Timer context manager
- Reproducibility (seed control)
- Saving / loading results
- Figure export helper
"""

import os
import gc
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# Randomness control for reproducibility
# ----------------------------------------

def set_seed(seed=42):
    """
    Set random seed across Python, NumPy, and (if available) Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"Random seed set to {seed}")


# ----------------------------------------
# Timing and logging helpers
# ----------------------------------------

class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        with Timer("Training Random Forest"):
            model.fit(X_train, y_train)
    """
    def __init__(self, name=""):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"[TIMER START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[TIMER END] {self.name} took {elapsed:.2f} seconds")


def log_message(message, logfile="results/logs/run_log.txt"):
    """
    Simple logger that appends timestamped messages to a text file.
    """
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(f"[LOG] {message}")


# ----------------------------------------
# Memory profiling utilities
# ----------------------------------------

def memory_usage_mb():
    """
    Return current Python process memory usage in MB.
    
    Uses gc.get_objects() to estimate memory consumption of all tracked objects.
    Note: This is an approximation and may underestimate actual memory usage.
    
    Returns:
        float: Estimated memory usage in megabytes
        
    Example:
        >>> print(f"Current memory: {memory_usage_mb():.2f} MB")
    """
    return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / 1024 / 1024


def print_memory_usage(label=""):
    """
    Print current memory usage with an optional label.
    
    Args:
        label (str): Optional descriptive label for the memory checkpoint
        
    Example:
        >>> print_memory_usage("After loading dataset")
        [MEMORY] After loading dataset: 1234.56 MB
    """
    mem_mb = memory_usage_mb()
    if label:
        print(f"[MEMORY] {label}: {mem_mb:.2f} MB")
    else:
        print(f"[MEMORY] Current usage: {mem_mb:.2f} MB")
    return mem_mb


def get_object_memory_usage(obj):
    """
    Get memory usage of a specific object in MB.
    
    Args:
        obj: Any Python object (DataFrame, array, list, etc.)
        
    Returns:
        float: Memory usage in megabytes
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': range(10000)})
        >>> print(f"DataFrame size: {get_object_memory_usage(df):.2f} MB")
    """
    # For pandas DataFrames, use memory_usage() method
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024 / 1024
    
    # For numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.nbytes / 1024 / 1024
    
    # For other objects, use sys.getsizeof
    return sys.getsizeof(obj) / 1024 / 1024


class MemoryTracker:
    """
    Context manager for tracking memory usage during code execution.
    
    Example:
        with MemoryTracker("Loading large dataset"):
            df = pd.read_csv('large_file.csv')
        # Output: [MEMORY] Loading large dataset - Delta: +123.45 MB (Start: 100.00 MB, End: 223.45 MB)
    """
    def __init__(self, name=""):
        self.name = name
        self.start_memory = None
        self.end_memory = None

    def __enter__(self):
        gc.collect()  # Force garbage collection before measuring
        self.start_memory = memory_usage_mb()
        print(f"[MEMORY START] {self.name}: {self.start_memory:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()  # Force garbage collection before measuring
        self.end_memory = memory_usage_mb()
        delta = self.end_memory - self.start_memory
        sign = "+" if delta >= 0 else ""
        print(f"[MEMORY END] {self.name} - Delta: {sign}{delta:.2f} MB "
              f"(Start: {self.start_memory:.2f} MB, End: {self.end_memory:.2f} MB)")


def cleanup_memory(*objects):
    """
    Delete objects and force garbage collection to free memory.
    
    Args:
        *objects: Variable number of objects to delete
        
    Example:
        >>> large_array = np.zeros((10000, 10000))
        >>> cleanup_memory(large_array)
        [MEMORY] Cleaned up 1 object(s), freed ~762.94 MB
    """
    mem_before = memory_usage_mb()
    
    for obj in objects:
        del obj
    
    gc.collect()
    mem_after = memory_usage_mb()
    freed = mem_before - mem_after
    
    print(f"[MEMORY] Cleaned up {len(objects)} object(s), freed ~{freed:.2f} MB")
    return freed


# ----------------------------------------
# Saving and loading helpers
# ----------------------------------------

def save_json(obj, filename):
    """
    Save Python object (dict, list, etc.) as a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)
    print(f"[SAVED] JSON -> {filename}")


def save_csv(df, filename):
    """
    Save pandas DataFrame as a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SAVED] DataFrame -> {filename}")


def load_json(filename):
    """
    Load JSON file into a Python object.
    """
    with open(filename, "r") as f:
        return json.load(f)


# ----------------------------------------
# Plotting utilities
# ----------------------------------------

def export_figure(fig, filename, dpi=300):
    """
    Save matplotlib figure with consistent formatting.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"[SAVED] Figure -> {filename}")


def plot_metric_bar(df, x_col, y_col, title, ylabel, filename=None):
    """
    Quick bar plot helper for model comparison charts.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df[x_col], df[y_col], color="skyblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    if filename:
        export_figure(fig, filename)
    else:
        plt.show()
    plt.close(fig)