# -*- coding: utf-8 -*-
"""Model v0: Tail-Aware Normal Feature Calibration."""
from .tail_adapter import ResidualAdapter
from .losses import (tail_loss, preserve_loss, relation_loss, mean_dist_loss,
                     separation_loss)

__all__ = ["ResidualAdapter", "tail_loss", "preserve_loss", "relation_loss",
           "mean_dist_loss", "separation_loss"]
