"""
Memory Management and Profiling Utilities for Navarasa AI.
Monitors process RSS, enforces garbage collection, and manages resource cleanup.
"""

import os
import gc
import sys
import logging
from typing import Optional, Dict, Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger("navarasa_memory")


def get_process_ram_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in Megabytes."""
    if HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            return 0.0
    return 0.0


def run_garbage_collection(tag: str = "") -> float:
    """
    Executes deep garbage collection, clears PyTorch cache if applicable,
    and returns current process RAM in MB.
    """
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    current_ram = get_process_ram_mb()
    if tag:
        logger.debug(f"[Memory Cleanup: {tag}] Process RAM: {current_ram:.2f} MB")
    return current_ram


# Alias for compatibility
get_process_memory_mb = get_process_ram_mb


class MemoryTracker:
    """Tracks memory progression across pipeline stages and computes peaks and deltas."""

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.history = []
        self.peak_mb = get_process_ram_mb()

    def checkpoint(self, stage_name: str) -> float:
        ram = get_process_ram_mb()
        self.history.append((stage_name, ram))
        if ram > self.peak_mb:
            self.peak_mb = ram
        return ram

    def print_summary(self):
        print(f"\n--- Memory Summary: {self.name} ---")
        for stage, ram in self.history:
            print(f"  {stage:<45}: {ram:7.2f} MB")
        print(f"  {'PEAK RAM':<45}: {self.peak_mb:7.2f} MB\n")

