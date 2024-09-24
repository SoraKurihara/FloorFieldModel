import os

from .cp import p_ij
from .dcm import L1norm, L2norm, Linfnorm
from .hc import handle_collisions
from .sql import create_sqlite, save_sqlite


def P():
  print("A")
  os.makedirs("test", exist_ok=True)
