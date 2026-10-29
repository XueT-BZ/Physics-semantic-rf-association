#!/usr/bin/env python3
"""Compatibility entry point for the revised result figures and tables."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('make_results_figures.py')),run_name='__main__')
