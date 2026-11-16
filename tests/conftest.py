"""Shared helper for generalized-coordinate checks."""

import torch


def identity_aux(batch_size: int) -> torch.Tensor:
  """Return a ``(batch_size, 7)`` aux state with identity quaternion."""
  aux = torch.zeros(batch_size, 7)
  aux[..., 3] = 1.0
  return aux
