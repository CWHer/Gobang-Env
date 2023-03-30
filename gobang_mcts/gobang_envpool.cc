#include "envpool/gobang_mcts/gobang_envpool.hpp"

#include "envpool/core/py_envpool.h"

using GobangEnvSpec = PyEnvSpec<GobangSpace::GobangEnvSpec>;
using GobangEnvPool = PyEnvPool<GobangSpace::GobangEnvPool>;

PYBIND11_MODULE(py_gobang_envpool, m) { REGISTER(m, GobangEnvSpec, GobangEnvPool) }
