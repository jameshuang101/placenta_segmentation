"""
Utility functions for dataset handling.
"""
import os
from typing import List

def list_files(directory: str, ext: str = '') -> List[str]:
    """
    List files in directory with given extension.
    """
    return [f for f in os.listdir(directory) if f.endswith(ext)]
