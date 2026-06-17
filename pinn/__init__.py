from pinn.model   import AcousticPINN, WaveEquationLoss
from pinn.sampler import (uniform_collocation, lhs_collocation,
                          boundary_points, initial_condition_points,
                          source_points, wall_clustered_collocation)
from pinn.trainer import PINNTrainer
