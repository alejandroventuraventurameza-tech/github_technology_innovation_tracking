"""
utils.py
--------
Shared utility functions used across the pipeline.

Track B — Technology Innovation & Ecosystem Tracking
"""

LABEL2ID = {
    "emerging": 0,
    "mature": 1,
    "declining": 2,
    "experimental": 3,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
