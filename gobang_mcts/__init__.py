from envpool.python.api import py_env

from .py_gobang_envpool import _GobangEnvSpec, _GobangEnvPool

GobangEnvSpec, GobangDMEnvPool, \
    GobangGymEnvPool, GobangGymnasiumEnvPool = py_env(
        _GobangEnvSpec, _GobangEnvPool
    )

__all__ = [
    "GobangEnvSpec",
    "GobangDMEnvPool",
    "GobangGymEnvPool",
    "GobangGymnasiumEnvPool",
]
