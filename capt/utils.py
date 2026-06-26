"""Small helpers for CAPT (vendored minimal subset, no external deps)."""
import numpy as np


def numeric_classes(tags_classes, dict_tags):
    """Map a list of string class tags to integer ids via dict_tags."""
    return np.array([dict_tags.get(t) for t in tags_classes])


def create_dict_texts(texts):
    """Build a {string: index} dict from a list of unique strings."""
    return {l: i for i, l in enumerate(sorted(set(texts)))}


def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]
