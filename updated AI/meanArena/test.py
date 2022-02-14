import mdptoolbox
import numpy as np

# Since the probability array is (A, S, S), there are 4 arrays. Each
# is a 4 x 4 array:
P2 = np.array([[[0.025, 0.95, 0.025, 0  ], # Right, State 0
                [0,   1,   0,   0  ], # State 1 is absorbing
                [0.025, 0,   0.025, 0.95],
                [0,   0,   0,   1  ]],# State 3 is absorbing
               [[0.9, 0,   0.1, 0  ], # Left
                [0,   1,   0,   0  ],
                [0.1, 0,   0.9, 0  ],
                [0,   0,   0,   1  ]],
               [[0.025, 0.025, 0.95, 0  ], # Up
                [0,   1,   0,   0  ],
                [0,   0,   0.9, 0.1],
                [0,   0,   0,   1  ]],
               [[0.9, 0.1, 0,   0  ], # Down
                [0,   1,   0,   0  ],
                [0.95, 0,   0.025, 0.025],
                [0,   0,   0,   1  ]]])

# The reward array has one set of values for each state. Each is the
# value of all the actions. Here there are four actions, all with the
# usual cost:
R2 = np.array([[-0.04, -0.04, -0.04, -0.04],
               [-1,    -1,    -1,    -1],
               [-0.04, -0.04, -0.04, -0.04],
               [1,      1,     1,     1]])

mdptoolbox.util.check(P2, R2)
vi2 = mdptoolbox.mdp.ValueIteration(P2, R2, 0.99)
vi2.run()
print('Values:\n', vi2.V)
print('Policy:\n', vi2.policy)
