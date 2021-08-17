import numpy as np
from copy import deepcopy
from numpy.random import RandomState
import logging
import torch as tch
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import os
from retina import Retina
import gym
from gym.utils import seeding
from gym import spaces
from networks import *
import json

environment_register = {
    'SnakePath':
        {
        'room_centers':
            np.array([
                [-2., -2.],
                [0., -2.],
                [2., -2.],
                [2., 0.],
                [0., 0.],
                [-2., 0.],
                [-2., 2.],
                [0., 2.],
                [2., 2.],
            ]),

        'room_sizes':
            np.array([
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
            ]),

        'room_exits':
            [ # (Exit_goes_to, door boundaries # exchange 3 and 5 for plot
                [{'goes_to': [1, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.}], # Room 0 exit
                [{'goes_to': [0, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
                 {'goes_to': [2, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 1 exits
                [{'goes_to': [1, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
                 {'goes_to': [3, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.},], # Room 2 exits

                [{'goes_to': [2, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y':-1.},
                 {'goes_to': [4, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.}, ], # Room 3 exits
                [{'goes_to': [3, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
                 {'goes_to': [5, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.}, ], # Room 4 exits
                [{'goes_to': [4, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
                 {'goes_to': [6, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.}, ], # Room 5 exits

                [{'goes_to': [5, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y':-1.},
                 {'goes_to': [7, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.}, ], # Room 6 exits
                [{'goes_to': [6, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
                 {'goes_to': [8, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.}, ], # Room 7 exits
                [{'goes_to': [7, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.}, ], # Room 8 exits
            ],

        'possible_reward_pos': {
            'None': {'room': 4, 'pos': [0., 0.]},
            'Default': {'room': 4, 'pos': [0., 0.]},
            'TopRight': {'room': 8, 'pos': [0., 0.]},
        },

        'possible_layouts':
            {'Default':
                [ # 'positions': (n_obj, 2) colors(n_obj, 3)
                    {'positions': np.array([
                        [0., 0.],
                        [2., 0.],
                        [4., 0.],
                        ]),
                     'colors': np.array([
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        ])
                    }, # Room 0


                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [2., 0],
                        ]),
                     'colors': np.array([
                        [0., 1., 0.],
                        [1., 0., 0.],
                        [0., 0., 1.],
                        ])
                    }, # Room 1

                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [-4., 0],
                        [0., 2.]
                        ]),
                     'colors': np.array([
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        ])
                    }, # Room 2

                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [-4., 0],
                        [0., -2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        [0., 0., 1.],
                        ])
                    }, # Room 3

                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [2., 0],
                        ]),
                     'colors': np.array([
                        [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        ])
                    }, # Room 4

                    {'positions': np.array([
                        [0, 0],
                        [2., 0.],
                        [4., 0.],
                        [0., 2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        ])
                    }, # Room 5

                    {'positions': np.array([
                        [0, 0],
                        [2.-1./2, 0.],
                        [2.+1./2, 0.],
                        [4., -1./2],
                        [4., 1./2],
                        [0., -2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        ])
                    }, # Room 6

                    {'positions': np.array([
                        [-1./2, 0],
                        [1./2, 0],
                        [-2., 0.],
                        [2., -1./2],
                        [2., +1./2],
                        ]),
                     'colors': np.array([
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        ])
                    }, # Room 7

                    {'positions': np.array([
                        [0., -1./2],
                        [0., 1./2],
                        [-4., 0],
                        [-2.-1./2, 0.],
                        [-2.+1./2, 0.],
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                     ])
                    }, # Room 8
                ],

            'DarkCenter':
                [ # 'positions': (n_obj, 2) colors(n_obj, 3)
                    {'positions': np.array([
                        [0., 0.],
                        [2., 0.],
                        [4., 0.],
                        ]),
                     'colors': np.array([
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        ])
                    }, # Room 0


                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [2., 0],
                        ]),
                     'colors': np.array([
                        [0., 1., 0.],
                        [1., 0., 0.],
                        [0., 0., 1.],
                        ])
                    }, # Room 1

                    {'positions': np.array([
                        [0, 0],
                        [-2., 0],
                        [-4., 0],
                        [0., 2.]
                        ]),
                     'colors': np.array([
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        ])
                    }, # Room 2

                    {'positions': np.array([
                        [0, 0],
                        # [-2., 0],
                        [-4., 0],
                        [0., -2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        # [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        [0., 0., 1.],
                        ])
                    }, # Room 3

                    {'positions': np.array([
                        [0, 0],
                        ]),
                     'colors': np.array([
                        [0, 0, 0],
                        ])
                    }, # Room 4

                    {'positions': np.array([
                        [0, 0],
                        # [2., 0.],
                        [4., 0.],
                        [0., 2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        # [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        ])
                    }, # Room 5

                    {'positions': np.array([
                        [0, 0],
                        [2.-1./2, 0.],
                        [2.+1./2, 0.],
                        [4., -1./2],
                        [4., 1./2],
                        [0., -2.]
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        [1./np.sqrt(2), 0, 1./np.sqrt(2)],
                        ])
                    }, # Room 6

                    {'positions': np.array([
                        [-1./2, 0],
                        [1./2, 0],
                        [-2., 0.],
                        [2., -1./2],
                        [2., +1./2],
                        ]),
                     'colors': np.array([
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        ])
                    }, # Room 7

                    {'positions': np.array([
                        [0., -1./2],
                        [0., 1./2],
                        [-4., 0],
                        [-2.-1./2, 0.],
                        [-2.+1./2, 0.],
                        ]),
                     'colors': np.array([
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1./np.sqrt(2), 0., 1./np.sqrt(2)],
                        [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                        [1., 0., 0.],
                        [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                     ])
                    }, # Room 8
                ]
        }
        },
    # 'DonutPath':
    #     {
    #     'room_centers':
    #         np.array([
    #             [-2., -2.],
    #             [0., -2.],
    #             [2., -2.],
    #             [2., 0.],
    #             [2., 2.],
    #             [0., 2.],
    #             [-2., 2.],
    #             [-2., 0.],
    #         ]),
    #
    #     'room_sizes':
    #         np.array([
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #             [1., 1.],
    #         ]),
    #
    #     'room_exits':
    #         [
    #             [{'goes_to': [1, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
    #              {'goes_to': [7, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.},], # Room 0 exits
    #             [{'goes_to': [2, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
    #               {'goes_to': [0, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 1 exits
    #             [{'goes_to': [3, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #               {'goes_to': [1, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 2 exits
    #             [{'goes_to': [4, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #               {'goes_to': [2, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 3 exits
    #             [{'goes_to': [5, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y': 0.},
    #               {'goes_to': [3, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 4 exits
    #             [{'goes_to': [6, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
    #                {'goes_to': [4, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 5 exits
    #             [{'goes_to': [7, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
    #               {'goes_to': [5, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},], # Room 6 exits
    #             [{'goes_to': [6, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                {'goes_to': [0, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 7 exits
    #         ],
    #     'possible_layouts':
    #         {'Default':
    #             [ # 'positions': (n_obj, 2) colors(n_obj, 3)
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [0., 2.],
    #                     [4., 0.],
    #                     [0., 4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                  ])
    #                 }, # Room 0
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [2., 0.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [0., 1., 0.],
    #                  ])
    #                 }, # Room 1
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [0., 2.],
    #                     [-4., 0.],
    #                     [0., 4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [0., 0., 1.],
    #                  ])
    #                  }, # Room 2
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [0., 2.],
    #                     [0., -2.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [0., 1., 0.],
    #                  ])
    #                 }, # Room 3
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [0., -2.],
    #                     [-4., 0.],
    #                     [0., -4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [0., 1., 0.],
    #                  ])
    #                  }, # Room 4
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [-2., 0.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                  ])
    #                 }, # Room 5
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [0., -2.],
    #                     [4., 0.],
    #                     [0., -4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [1., 0., 1.],
    #                  ])
    #                  }, # Room 6
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [0., 2.],
    #                     [0., -2.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1., 0., 0.],
    #                  ])
    #                 }, # Room 7
    #             ]
    #         },
    #     },

    #     'DoubleDonut':
    #         {
    #         'room_centers':
    #             np.array([
    #                 # 1st donut (left)
    #                 [-2., -2., 0],
    #                 [0., -2., 0],
    #                 [2., -2., 0],
    #                 [2., 0., 0],
    #                 [2., 2., 0],
    #                 [0., 2., 0],
    #                 [-2., 2., 0],
    #                 [-2., 0., 0],
    #
    #                 # 2nd donut (right)
    #                 [4., -2., 0],
    #                 [6., -2., 0],
    #                 [8., -2., 0],
    #                 [8., 0., 0],
    #                 [8., 2., 0],
    #                 [6., 2., 0],
    #                 [4., 2., 0],
    #                 [4., 0., 0],
    #
    #             ]),
    #
    #         'room_sizes':
    #             np.array([
    #                 # 1st floor
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #
    #                 # 2nd floor
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #                 [1., 1.],
    #             ]),
    #
    #         'room_exits':
    #             [ # Have to add exits only to 2, 3, 4
    #                 [{'goes_to': [1, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},
    #                  {'goes_to': [7, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.}, ], # Room 0 exits
    #                 [{'goes_to': [2, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
    #                   {'goes_to': [0, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 1 exits
    #                 [{'goes_to': [3, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                   {'goes_to': [1, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
    #                   {'goes_to': [8, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 2 exits
    #                 [{'goes_to': [4, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                   {'goes_to': [2, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
    #                   {'goes_to': [15, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 3 exits
    #                 [{'goes_to': [5, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y': 0.},
    #                   {'goes_to': [3, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
    #                   {'goes_to': [14, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 4 exits
    #                 [{'goes_to': [6, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
    #                    {'goes_to': [4, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 5 exits
    #                 [{'goes_to': [7, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
    #                   {'goes_to': [5, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},], # Room 6 exits
    #                 [{'goes_to': [6, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                    {'goes_to': [0, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 7 exits
    #
    #                   # Add only to 8, 14, 15
    #                 [{'goes_to': [9, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
    #                 {'goes_to': [15, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.},
    #                 {'goes_to': [2, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 8 exits
    #                 [{'goes_to': [10, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
    #                   {'goes_to': [8, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 9 exits
    #                 [{'goes_to': [11, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                   {'goes_to': [9, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 10 exits
    #                 [{'goes_to': [12, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                   {'goes_to': [10, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 11 exits
    #                 [{'goes_to': [13, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y': 0.},
    #                   {'goes_to': [11, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 12 exits
    #                 [{'goes_to': [14, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
    #                    {'goes_to': [12, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 13 exits
    #                 [{'goes_to': [15, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
    #                   {'goes_to': [13, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},
    #                   {'goes_to': [4, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 14 exits
    #                 [{'goes_to': [14, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
    #                 {'goes_to': [3, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
    #                 {'goes_to': [8, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 15 exits
    #             ],
    #         'possible_layouts':
    #         # For right donut, keep the same corner colors but double the objects (y =  +- .5)
    #
    #             {'Default':
    #                 [ # 'positions': (n_obj, 2) colors(n_obj, 3)
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [2., 0.],
    #                         [0., 2.],
    #                         [4., 0.],
    #                         [0., 4.],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1., 0., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                      ])
    #                     }, # Room 0
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [-2., 0.],
    #                         [2., 0.],
    #                         [4., .5],
    #                         [4., -.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1., 0., 0.],
    #                         [0., 1., 0.],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                      ])
    #                     }, # Room 1
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [-2., 0.],
    #                         [0., 2.],
    #                         [-4., 0.],
    #                         [0., 4.],
    #                         [2., .5],
    #                         [2., -.5],
    #                         [4., .5],
    #                         [4., -.5],
    #                         [2., 1.5],
    #                         [2., 2.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1., 0., 0.],
    #                         [0., 0., 1.],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                      ])
    #                      }, # Room 2
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [0., 2.],
    #                         [0., -2.],
    #                         [2., .5],
    #                         [2., -.5],
    #                         [2., 1.5],
    #                         [2., 2.5],
    #                         [2., -1.5],
    #                         [2., -2.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                      ])
    #                     }, # Room 3
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [-2., 0.],
    #                         [0., -2.],
    #                         [-4., 0.],
    #                         [0., -4.],
    #                         [2., .5],
    #                         [2., -.5],
    #                         [4., .5],
    #                         [4., -.5],
    #                         [2., -1.5],
    #                         [2., -2.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                      ])
    #                      }, # Room 4
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [2., 0.],
    #                         [-2., 0.],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                      ])
    #                     }, # Room 5
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [2., 0.],
    #                         [0., -2.],
    #                         [4., 0.],
    #                         [0., -4.],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [1., 0., 1.],
    #                      ])
    #                      }, # Room 6
    #
    #                     {'positions': np.array([
    #                         [0., 0.],
    #                         [0., 2.],
    #                         [0., -2],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1., 0., 0.],
    #                      ])
    #                     }, # Room 7
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [2., 0.5],
    #                         [2., -0.5],
    #                         [4, 0.5],
    #                         [4, -0.5],
    #                         [0., 1.5],
    #                         [0., 2.5],
    #                         [0., 3.5],
    #                         [0., 4.5],
    #                         [-2, 0.],
    #                         [-4, 0.],
    #                         [-2., 2.]
    #                                  ]),
    #                      'colors': np.array([
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                      ])
    #                     }, # Room 8
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [-2., -0.5],
    #                         [-2., 0.5],
    #                         [2., 0.5],
    #                         [2., -0.5],
    #                         [-4., 0.]
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                      ])
    #                     }, # Room 9
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [-2., 0.5],
    #                         [-2., -0.5],
    #                         [0., 2.5],
    #                         [0., 1.5],
    #                         [-4., 0.5],
    #                         [-4., -0.5],
    #                         [0., 4.5],
    #                         [0., 3.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                      ])
    #                      }, # Room 10
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [0., 2.5],
    #                         [0., 1.5],
    #                         [0., -1.5],
    #                         [0., -2.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                      ])
    #                     }, # Room 11
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [-2., 0.5],
    #                         [-2., -0.5],
    #                         [0., -2.5],
    #                         [0., -1.5],
    #                         [-4., 0.5],
    #                         [-4., -0.5],
    #                         [0., -3.5],
    #                         [0., -4.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [0., 1., 0.],
    #                         [0., 1., 0.],
    #                      ])
    #                      }, # Room 12
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [2., 0.5],
    #                         [2., -0.5],
    #                         [-2., 0.5],
    #                         [-2., -0.5],
    #                         [-4., 0.],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [0., 0., 1.],
    #                      ])
    #                     }, # Room 13
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [2., 0.5],
    #                         [2., -0.5],
    #                         [0., -2.5],
    #                         [0., -1.5],
    #                         [4., 0.5],
    #                         [4., -0.5],
    #                         [-2., 0.],
    #                         [-4., 0.],
    #                         [-2., -2.],
    #                         [0., -3.5],
    #                         [0., -4.5],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                      ])
    #                      }, # Room 14
    #
    #                     {'positions': np.array([
    #                         [0., 0.5],
    #                         [0., -0.5],
    #                         [0., 1.5],
    #                         [0., 2.5],
    #                         [0., -1.5],
    #                         [0., -2.5],
    #                         [-2., 2.],
    #                         [-2., 0],
    #                         [-2., -2],
    #                                  ]),
    #                      'colors': np.array([
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                         [1., 0., 0.],
    #                         [1., 0., 0.],
    #                         [0., 0., 1.],
    #                         [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                         [0., 1., 0.],
    #                      ])
    #                     }, # Room 15
    #
    #                 ],
    #
    #             # Make the two opposite ends of the "infinite" identical, so that 7 and 11 are ambiguous, force no resetting
    #             'Ambiguous': [ # 'positions': (n_obj, 2) colors (n_obj, 3)
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [0., 2.],
    #                     [4., 0.],
    #                     [0., 4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                  ])
    #                 }, # Room 0
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [2., 0.],
    #                     [4., .5],
    #                     [4., -.5],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [0., 1., 0.],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                  ])
    #                 }, # Room 1
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [0., 2.],
    #                     [-4., 0.],
    #                     [0., 4.],
    #                     [2., .5],
    #                     [2., -.5],
    #                     [4., .5],
    #                     [4., -.5],
    #                     [2., 1.5],
    #                     [2., 2.5],
    #                              ]),
    #                  'colors': np.array([
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [0., 0., 1.],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                  ])
    #                  }, # Room 2
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [0., 2.],
    #                     [0., -2.],
    #                     [2., .5],
    #                     [2., -.5],
    #                     [2., 1.5],
    #                     [2., 2.5],
    #                     [2., -1.5],
    #                     [2., -2.5],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                  ])
    #                 }, # Room 3
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.],
    #                     [0., -2.],
    #                     [-4., 0.],
    #                     [0., -4.],
    #                     [2., .5],
    #                     [2., -.5],
    #                     [4., .5],
    #                     [4., -.5],
    #                     [2., -1.5],
    #                     [2., -2.5],
    #                              ]),
    #                  'colors': np.array([
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                  ])
    #                  }, # Room 4
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [-2., 0.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                  ])
    #                 }, # Room 5
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [2., 0.],
    #                     [0., -2.],
    #                     [4., 0.],
    #                     [0., -4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 0., 1.],
    #                     [1., 0., 1.],
    #                  ])
    #                  }, # Room 6
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [0., 2.],
    #                     [0., -2],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1., 0., 0.],
    #                  ])
    #                 }, # Room 7
    #
    #                 {'positions': np.array([
    #                     [0., 0.5],
    #                     [0., -0.5],
    #                     [2., 0.5],
    #                     [2., -0.5],
    #                     [4, 0.],
    #                     [0., 1.5],
    #                     [0., 2.5],
    #                     [0., 3.5],
    #                     [0., 4.5],
    #                     [-2, 0.],
    #                     [-4, 0.],
    #                     [-2., 2.]
    #                              ]),
    #                  'colors': np.array([
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [0., 1., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                  ])
    #                 }, # Room 8
    #
    #                 {'positions': np.array([
    #                     [0., 0.5],
    #                     [0., -0.5],
    #                     [-2., -0.5],
    #                     [-2., 0.5],
    #                     [2., 0.],
    #                     [-4., 0.]
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [0., 1., 0.],
    #                  ])
    #                 }, # Room 9
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.5],
    #                     [-2., -0.5],
    #                     [0., 2],
    #                     [-4., 0.5],
    #                     [-4., -0.5],
    #                     [0., 4.],
    #
    #                              ]),
    #                  'colors': np.array([
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                  ])
    #                  }, # Room 10
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [0., 2.],
    #                     [0., -2.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1., 0., 0.],
    #
    #                  ])
    #                 }, # Room 11
    #
    #                 {'positions': np.array([
    #                     [0., 0.],
    #                     [-2., 0.5],
    #                     [-2., -0.5],
    #                     [0., -2.],
    #                     [-4., 0.5],
    #                     [-4., -0.5],
    #                     [0., -4.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1., 0., 0.],
    #                  ])
    #                  }, # Room 12
    #
    #                 {'positions': np.array([
    #                     [0., 0.5],
    #                     [0., -0.5],
    #                     [2., 0.],
    #                     [-2., 0.5],
    #                     [-2., -0.5],
    #                     [-4., 0.],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [0., 0., 1.],
    #                  ])
    #                 }, # Room 13
    #
    #                 {'positions': np.array([
    #                     [0., 0.5],
    #                     [0., -0.5],
    #                     [2., 0.5],
    #                     [2., -0.5],
    #                     [0., -2.5],
    #                     [0., -1.5],
    #                     [4., 0.],
    #                     [-2., 0.],
    #                     [-4., 0.],
    #                     [-2., -2.],
    #                     [0., -3.5],
    #                     [0., -4.5],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                  ])
    #                  }, # Room 14
    #
    #                 {'positions': np.array([
    #                     [0., 0.5],
    #                     [0., -0.5],
    #                     [0., 1.5],
    #                     [0., 2.5],
    #                     [0., -1.5],
    #                     [0., -2.5],
    #                     [-2., 2.],
    #                     [-2., 0],
    #                     [-2., -2],
    #                              ]),
    #                  'colors': np.array([
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1./np.sqrt(2), 1./np.sqrt(2), 0.],
    #                     [1., 0., 0.],
    #                     [1., 0., 0.],
    #                     [0., 0., 1.],
    #                     [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    #                     [0., 1., 0.],
    #                  ])
    #                 }, # Room 15
    #
    #             ],
    #             },
    # },

}


# These are made to use steps of size .5*env.scale, ie 1/4 of a room each time
meaningful_trajectories = {
    'SnakePath': np.array([
                [-.5, .5],
                [0, -.5],
                [0, -.5],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [0., .5],
                [0., .5],
                [0., .5],
                [0., .5],
                [0., .5],
                [0., .5],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [0., .5],
                [0., .5],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [0., .5],
                [0., .5],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [0., -.5],
                [0., -.5],
                [0., -.5],
                [0., -.5],
                [0., -.5],
                [0., -.5],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [.5, 0.],
                [0., -.5],
                [0., -.5],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                [-.5, 0.],
                ]),




    'DoubleDonut': np.array([
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.45],
                                [0., -.45],
                                [0., -.45],
                                [0., -.45],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                # Loop around first donut

                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                # Enter second one

                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [.5, 0.],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                [0., -.5],
                                # Loop around second donut

                                [-.45, 0.05],
                                [-.45, 0.05],
                                [-.45, 0.05],
                                [-.45, 0.05],
                                # Back to first

                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [.5, 0.05],
                                [.5, 0.05],
                                [.5, 0.05],
                                [.5, 0.05],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                [0., .5],
                                # Move back and forth

                                [.5, 0.05],
                                [.5, 0.05],
                                [.5, 0.05],
                                [.5, 0.05],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],
                                [-.5, 0.],

                              ])
}

# TODO: after implementing ResetNetwork.step, modify env to store the internal state of the RNN (initialized in reset) and update it after each action; use that as preprocessed image in get_observation

class World:
    def __init__(self, map_name='SnakePath', scale=.5, reward_area_width=.3, objects_layout_name='Default', epoch_len=20, seed=0, load_preprocessor_from=None, **kwargs):
        self.epoch_len = epoch_len
        self.retina = Retina(n=64**2, bounds=[-.5,.5], widths=[.3, .5], device_name='cuda')
        self.device = self.retina.device
        self.map_name = map_name
        self.objects_layout_name = objects_layout_name

        env_params = deepcopy(environment_register[map_name])
        self.env_params = env_params
        self.n_rooms = len(env_params['room_centers'])
        logging.critical('n_rooms in our env : {}'.format(self.n_rooms))
        self.room_labels = ["Room {}".format(i) for i in range(self.n_rooms)]
        self.room_centers = env_params['room_centers']
        self.room_sizes = env_params['room_sizes']
        self.room_exits = env_params['room_exits']
        self.room_objects = env_params['possible_layouts'][objects_layout_name]
        self.scale = scale

        self.max_objects_per_room = np.max([objs_dict['positions'].shape[0] for objs_dict in self.room_objects])
        self.colors_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 3))
        self.positions_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 2))
        for room_idx in range(self.n_rooms):
            for obj_idx in range(self.room_objects[room_idx]['colors'].shape[0]):
                self.colors_blob[room_idx, obj_idx] = self.room_objects[room_idx]['colors'][obj_idx]
                self.positions_blob[room_idx, obj_idx] = self.room_objects[room_idx]['positions'][obj_idx]

        self.room_sizes = scale * self.room_sizes
        self.room_centers = scale * self.room_centers

        for i in range(self.n_rooms):
            self.room_objects[i]['positions'] = scale * self.room_objects[i]['positions']
            for obj_idx in range(len(self.room_exits[i])):
                self.room_exits[i][obj_idx]['goes_to'][1] = scale * np.array(self.room_exits[i][obj_idx]['goes_to'][1])
                self.room_exits[i][obj_idx]['x'] = scale * self.room_exits[i][obj_idx]['x']
                self.room_exits[i][obj_idx]['y'] = scale * self.room_exits[i][obj_idx]['y']

        # Useful for losses; for now, only rooms with no objects
        self.rooms_not_to_start_in = [i for i in range(self.n_rooms) if np.all(self.room_objects[i]['colors'] == 0.)]
        self.possible_start_rooms = [i for i in range(self.n_rooms) if not np.all(self.room_objects[i]['colors'] == 0.)]

        if self.map_name == 'DoubleDonut' and self.objects_layout_name == 'Ambiguous':
             self.possible_start_rooms = np.array([0,1,2,3,4,5,6,8,9,10,12,13,14,15]) # Do not allow start from ambiguous rooms

        logging.critical('Allowed starting rooms : {}'.format(self.possible_start_rooms))
        self.set_seed(seed)

        self.seed_value = seed

        if load_preprocessor_from is not None:
            with open(load_preprocessor_from+'/full_params.json', mode='r') as f:
                params = json.load(f)['net_params']['options']
            self.preprocessor = BigResetNetwork(**params).to(self.device)
            logging.critical('Attempting to load preprocessor from : {}'.format(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value)))
            self.preprocessor.load(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value))
        else:
            self.preprocessor = None


        # Make sure that everything is normalized reasonably
        # rooms = np.random.choice(self.possible_start_rooms, size=(1024, ))
        # xy_room = np.random.uniform(-self.scale, self.scale, size=(1024, 2))
        # images = self.get_images(rooms, xy_room)
        # logging.critical('Shape of images : {}'.format(images.shape))
        # norms = tch.sqrt(tch.mean((images.view(images.shape[0], -1)**2), dim=-1))
        # logging.critical('Norms : {}'.format(norms))
        # plt.figure()
        # norms = norms.cpu().numpy()
        # plt.hist(norms)
        # plt.show()
        #
        # plt.figure()
        # plop = images.view(images.shape[0], -1).cpu().numpy()
        # maxs = plop.max(axis=-1)
        # plt.hist(plop.flatten())
        # # plt.hist(maxs)
        # plt.show()

    def set_seed(self, seed=None):
        logging.critical('called world.seed')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render_background(self, ax_to_use=None):
        if ax_to_use is None:
            fig, ax = plt.subplots()
        else:
            ax = ax_to_use

        ax.set_facecolor([0,0,0,0])
        ax.set_axis_off()
        ax.set_aspect('equal')

        ax.set_xlim([np.min(self.room_centers[:,0])-self.scale-.05, np.max(self.room_centers[:,0])+self.scale+.05])
        ax.set_ylim([np.min(self.room_centers[:,1])-self.scale-.05, np.max(self.room_centers[:,1])+self.scale+.05])

        # This one is common to all environments, it determines the enclosing area
        rect = patches.Rectangle((np.min(self.room_centers[:,0])-self.scale, np.min(self.room_centers[:,1]-self.scale)),
                                np.max(self.room_centers[:,0]) - np.min(self.room_centers[:,0])+ 2*self.scale,
                                np.max(self.room_centers[:,1]) - np.min(self.room_centers[:,1])+ 2*self.scale,
                                linewidth=1, edgecolor='k',facecolor=[0,0,0,0])
        ax.add_patch(rect)

        if self.map_name == 'SnakePath':
            ax.plot([-3 *self.scale, self.scale], [-self.scale, -self.scale], c='k', linewidth=3)
            ax.plot([-self.scale, 3*self.scale], [self.scale, self.scale], c='k', linewidth=3)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DonutPath':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DoubleDonut':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

            rect = patches.Rectangle((5*self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([3*self.scale, 3*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([5*self.scale, 5*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([7*self.scale, 7*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

        if ax_to_use is None:
            return fig, ax
        else:
            return ax

    def get_observation(self, room, position):
        position = tch.from_numpy(self.positions_blob[room] - position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0)).float()
        color = tch.from_numpy(self.colors_blob[room]).float().to(self.retina.device)
        image_batch = self.retina.activity(position, color)
        if self.preprocessor is not None:
            with tch.set_grad_enabled(False):
                image_batch = self.preprocessor.get_representation(image_batch)
            image_batch = image_batch.view(self.preprocessor.representation_size).detach().cpu().numpy()
        else:
            image_batch = image_batch.squeeze(0).detach().cpu().numpy()
        return image_batch

    def get_images(self, rooms, positions):
        # logging.critical('In New World : rooms {} ; positions {}'.format(rooms[0, :3], positions[0, :3]))


        desired_shape = deepcopy(rooms.shape)
        rooms = rooms.flatten().astype(int)
        positions = positions.reshape(-1, 2)
        # logging.critical('positions info : shape {}, type {}'.format(positions.shape, positions.dtype))
        all_positions = tch.from_numpy(self.positions_blob[rooms] - positions[:, np.newaxis, :].repeat(self.max_objects_per_room, axis=1)).float()

        all_positions = all_positions.to(self.retina.device)
        all_colors = tch.from_numpy(self.colors_blob[rooms]).float().to(self.retina.device)
        # logging.critical('In new World : allpositions {} ; all_colors {}'.format(all_positions[:3], all_colors[:3]))
        image_batch = self.retina.activity(all_positions, all_colors)

        return image_batch.reshape(*desired_shape, self.retina.n, 3)#.cpu().numpy()

    def check_reward_overlap(self, room, pos):
        overlaps = np.logical_and(room == self.reward_room, np.max(np.abs(pos-self.reward_pos))<=self.reward_area_width)
        return overlaps

    # def reset(self):
    #     self.t = 0
    #     self.agent_room = self.np_random.choice(self.possible_start_rooms)
    #     Lx, Ly = self.room_sizes[self.agent_room]
    #     self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
    #     obs = self.get_observation(self.agent_room, self.agent_position)
    #     # assert self.observation_space.contains(obs)
    #     return obs

    # Below that are functions used for legacy interface
    def set_agent_position(self, room, xy=(0,0)):
        self.agent_room = room
        Lx, Ly = self.room_sizes[self.agent_room]
        invalid_x = xy[0]>Lx or xy[0]<-Lx
        invalid_y = xy[1]>Ly or xy[1]<-Ly
        if invalid_x or invalid_y:
            raise RuntimeError('Invalid xy initialization for current room')
        self.agent_position = np.array([*xy])

    def render_template(self, ax_to_use=None):
        if ax_to_use is None:
            fig, ax = self.render_background(ax_to_use=None)
        else:
            ax = self.render_background(ax_to_use=ax_to_use)

        for room in range(self.n_rooms):
            for xy, col in zip(self.room_objects[room]['positions'], self.room_objects[room]['colors']):
                xy0 = self.room_centers[room, :2]
                rect = patches.Rectangle(xy+xy0-.05*self.scale, .2*self.scale, .2*self.scale, linewidth=1, edgecolor='k', facecolor=col)
                ax.add_patch(rect)

        exit_rect = patches.Rectangle(self.room_centers[self.reward_room, :2]+self.reward_pos-self.reward_area_width, 2*self.reward_area_width, 2*self.reward_area_width, facecolor='gray', hatch='x', alpha=.5)
        ax.add_patch(exit_rect)

        if ax_to_use is None:
            return fig, ax
        else:
            return ax




    def __replay_one_traj(self, actions, start_room=None, start_pos=None):
        self.reset()
        epoch_len = actions.shape[0]
        positions = np.zeros((epoch_len+1,2))
        validated_actions = np.zeros((epoch_len,2))
        rooms = np.zeros((epoch_len+1))

        if start_room is None:
            room, pos = self.agent_room, self.agent_position
        else:
            room, pos = start_room, start_pos
            room = int(room)
            self.set_agent_position(room, (pos[0], pos[1]))

        positions[0] = pos
        rooms[0] = room

        for idx, action in enumerate(actions):
            # new_room, new_pos, rectified_action, reward, is_terminal = self.step(action)
            obs, reward, end_traj, info = self.step(action)
            new_room, new_pos, rectified_action = info['new_room'], info['new_pos'], info['rectified_action']
            logging.debug('Start in room {} at ({},{}) and end in room {} at ({},{}) with tot_dep ({},{})'.format(room, *pos, new_room, *new_pos, *rectified_action))
            validated_actions[idx] = rectified_action
            positions[idx+1] = new_pos
            rooms[idx+1] = new_room
            pos = new_pos
            room = new_room

            if end_traj:
                positions[idx+1:] = new_pos
                rooms[idx+1:] = new_room
                validated_actions[idx+1:] = 0.
                break


        return rooms, positions, validated_actions


    def static_replay(self, actions_batch, start_rooms=None, start_pos=None):
        actions_batch_local = deepcopy(actions_batch)
        batch_size = actions_batch.shape[0]
        epoch_len = actions_batch.shape[1]

        if start_rooms is not None:
            assert start_pos is not None
            assert start_rooms.shape[0] == start_pos.shape[0]
            assert start_pos.shape[1] == 2

        rooms = np.zeros((batch_size, epoch_len+1))
        positions = np.zeros((batch_size, epoch_len + 1, 2))
        validated_actions = np.zeros((batch_size, epoch_len, 2))
        rewards = np.zeros((batch_size, epoch_len+1))
        irrelevant_times = np.zeros((batch_size, epoch_len+1))

        # NOTE: making this multi-threaded seems smart, but at least in my tests its either slow, buggy, or both
        for b in range(batch_size):
            if start_rooms is None:
                room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
            else:
                room, pos, act, =  self.__replay_one_traj(actions_batch_local[b], start_room=start_rooms[b], start_pos=start_pos[b])

            rooms[b, :] = room
            positions[b, :] = pos
            validated_actions[b, :] = act

        logging.debug('Done with static_replay in environments.py')
        return rooms, positions, validated_actions


    @staticmethod
    def __add_arrows(line, size=15, color=None, zorder=-1):

        if color is None:
            color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        x_ends = .5* (xdata[:-1] + xdata[1:])
        y_ends = .5* (ydata[:-1] + ydata[1:])

        for x_start, x_end, y_start, y_end in zip(xdata, x_ends, ydata, y_ends):
            line.axes.annotate('',
                xytext=(x_start, y_start),
                xy=(x_end, y_end),
                arrowprops=dict(arrowstyle="->", color=color),
                size=size, zorder=-1
            )


    def plot_trajectory(self, actions, start_room=None, start_pos=None, ax_to_use=None, save_file=None, color=None, marker='+', zorder=500, show_lines=False, show_arrows=False, s=32, **kwargs):
        # By default, color is set to show time, but can be overridden
        if color is None:
            time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[0]+1)
            cmap = plt.get_cmap('jet')
            color = cmap(time_based_norm(range(actions.shape[0]+1)))

        if ax_to_use is None:
            if self.map_name != 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(5,5))
            elif self.map_name == 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(10,5))

        else:
            ax = ax_to_use

        if start_room is None:
            room, pos, act =  self.__replay_one_traj(actions, start_room=None, start_pos=None)
        else:
            room, pos, act =  self.__replay_one_traj(actions, start_room=start_room, start_pos=start_pos)

        ax = self.render_template(ax_to_use=ax)
        absolute_pos = pos+self.room_centers[room.astype(int), :2]

        ax.scatter(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, marker=marker, s=s)

        if show_lines:
            lines = ax.plot(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, ls='-')
            if show_arrows:
                self.__add_arrows(lines[0], zorder=zorder, color=color)

        if save_file is not None:
            os.makedirs('/'.join(save_file.split('/')[:-1]), exist_ok=True)
            fig.savefig(save_file)
            plt.close(fig)
        else:
            return ax

    def movement_logic(self, action):
        assert len(action) == 2
        action_bkp = deepcopy(action)
        bkp_x, bkp_y = deepcopy(self.agent_position)
        room_bkp = deepcopy(self.agent_room)

        new_pos = np.array([self.agent_position[0] + action[0], self.agent_position[1] + action[1]])
        Lx, Ly = self.room_sizes[self.agent_room]

        invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx
        invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly

        if invalid_x and invalid_y:
            if self.np_random.uniform() < .5:
                action[0] -= new_pos[0] - np.clip(new_pos[0], -Lx, Lx)
                new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
                invalid_x = False
            else:
                action[1] -= new_pos[1] - np.clip(new_pos[1], -Ly, Ly)
                new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
                invalid_y = False


        changed_room = False
        if invalid_y:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='horizontal':
                    if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
                        logging.debug('crossed horizontal door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if invalid_x:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='vertical':
                    if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
                        logging.debug('crossed vertical door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if not changed_room:
            new_room = room_bkp
        else:
            new_room = used_exit['goes_to'][0]
            new_pos = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[new_room, :2]

        rectified_new_pos = np.zeros(2)
        rectified_new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
        rectified_new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
        rectified_action = action + rectified_new_pos - new_pos

        self.agent_room = deepcopy(new_room)
        self.agent_position = deepcopy(rectified_new_pos)

        self.t += 1
        return self.agent_room, self.agent_position, deepcopy(rectified_action)

    # def step(self, action):
    #     rectified_action = self.movement_logic(action)
    #     return deepcopy(self.agent_room), deepcopy(self.agent_position), rectified_action



class FixedRewardWorld(gym.Env, World):
    def __init__(self, reward_area_width=.3, chosen_reward_pos='Default', epoch_len=100, **kwargs):
        World.__init__(self, **kwargs)
        logging.critical(kwargs)

        self.epoch_len = epoch_len
        self.reward_room = self.env_params['possible_reward_pos'][chosen_reward_pos]['room']
        self.reward_pos = np.array(self.env_params['possible_reward_pos'][chosen_reward_pos]['pos']) * self.scale
        self.reward_area_width = reward_area_width * self.scale

        # Gym specific stuff
        self.action_space = gym.spaces.Box(low=.8*self.scale *np.array([-1.0, -1.0]), high=.8*self.scale *np.array([1.0, 1.0]), dtype=np.float32) # Continuous actions, bounded for simplicity
        if self.preprocessor is None:
            self.observation_shape = (64**2, 3)
        else:
            self.observation_shape = (self.preprocessor.representation_size,)

        self.observation_space = spaces.Box(low = -100 * np.ones(self.observation_shape),
                                            high = 100 * np.ones(self.observation_shape),
                                            dtype = np.float32)
        # print(self.__dict__.keys())

    def reset(self):
        self.t = 0
        overlap_exit = True

        while overlap_exit:
            self.agent_room = self.np_random.choice(self.possible_start_rooms)
            Lx, Ly = self.room_sizes[self.agent_room]
            self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
            overlap_exit = self.check_reward_overlap(self.agent_room, self.agent_position)

        obs = self.get_observation(self.agent_room, self.agent_position)

        assert self.observation_space.contains(obs)
        return obs


    def step(self, action):
        assert len(action) == 2
        new_room, rectified_new_pos, rectified_action = self.movement_logic(action)

        # Decouple those now, could be useful later
        end_traj = self.check_reward_overlap(self.agent_room, self.agent_position)
        reward = self.check_reward_overlap(self.agent_room, self.agent_position).astype(np.float32)

        if not np.all(rectified_action == action):
            reward = -0.05
        if reward == 0.:
            reward = -0.01

        info = {'new_room': deepcopy(new_room), 'new_pos': deepcopy(rectified_new_pos), 'rectified_action': deepcopy(rectified_action)}

        if self.t >= self.epoch_len:
            end_traj = True
        obs = self.get_observation(self.agent_room, self.agent_position)
        assert self.observation_space.contains(obs)
        return obs, reward, end_traj, info



class GoalBasedWorld(gym.GoalEnv, FixedRewardWorld):
    def __init__(self,  **kwargs):
        FixedRewardWorld.__init__(self, **kwargs)
        self.reset()


    def reset(self):
        # logging.critical('reset called')
        self.t = 0

        self.reward_room = self.np_random.choice(self.possible_start_rooms)
        Lx, Ly = self.room_sizes[self.reward_room]
        self.reward_position = np.array([self.np_random.uniform(-Lx+self.reward_area_width, Lx-self.reward_area_width), self.np_random.uniform(-Ly+self.reward_area_width, Ly-self.reward_area_width)])


        overlap_exit = True
        while overlap_exit:
            self.agent_room = self.np_random.choice(self.possible_start_rooms)
            Lx, Ly = self.room_sizes[self.agent_room]
            self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
            overlap_exit = self.check_reward_overlap(self.agent_room, self.agent_position)

        obs = self.get_observation(self.agent_room, self.agent_position)
        goal_obs = self.get_observation(self.reward_room, self.reward_position)

        self.goal_obs = deepcopy(goal_obs)

        return {'observation': obs, 'desired_goal': goal_obs, 'achieved_goal': obs}


# Variants (for-v1, v2, etc...)
class FixedRewardPreprocessedWorld(FixedRewardWorld):
    def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', chosen_reward_pos='Default', epoch_len=20, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/'):
        FixedRewardWorld.__init__(self, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, chosen_reward_pos=chosen_reward_pos, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from)

class GoalBasedPreprocessedWorld(GoalBasedWorld):
    def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', epoch_len=30, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/'):
        GoalBasedWorld.__init__(self, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from)

















class LegacyWorld:
    def __init__(self, map_name='BigRoom', scale=.5,  objects_layout_name='SingleCentered', seed=0, **kwargs):
        self.seed = seed
        self.retina = Retina(n=64**2, bounds=[-.5,.5], widths=[.3, .5], device_name='cuda')
        self.rng = RandomState(seed)
        self.map_name = map_name
        self.objects_layout_name = objects_layout_name

        env_params = deepcopy(environment_register[map_name])
        self.n_rooms = len(env_params['room_centers'])
        logging.critical('n_rooms in our env : {}'.format(self.n_rooms))
        self.room_labels = ["Room {}".format(i) for i in range(self.n_rooms)]
        self.room_centers = env_params['room_centers']
        self.room_sizes = env_params['room_sizes']
        self.room_exits = env_params['room_exits']
        self.room_objects = env_params['possible_layouts'][objects_layout_name]
        self.scale = scale

        self.max_objects_per_room = np.max([objs_dict['positions'].shape[0] for objs_dict in self.room_objects])
        self.colors_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 3))
        self.positions_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 2))
        for room_idx in range(self.n_rooms):
            for obj_idx in range(self.room_objects[room_idx]['colors'].shape[0]):
                self.colors_blob[room_idx, obj_idx] = self.room_objects[room_idx]['colors'][obj_idx]
                self.positions_blob[room_idx, obj_idx] = self.room_objects[room_idx]['positions'][obj_idx]

        self.room_sizes = scale * self.room_sizes
        self.room_centers = scale * self.room_centers

        for i in range(self.n_rooms):
            self.room_objects[i]['positions'] = scale * self.room_objects[i]['positions']
            for obj_idx in range(len(self.room_exits[i])):
                self.room_exits[i][obj_idx]['goes_to'][1] = scale * np.array(self.room_exits[i][obj_idx]['goes_to'][1])
                self.room_exits[i][obj_idx]['x'] = scale * self.room_exits[i][obj_idx]['x']
                self.room_exits[i][obj_idx]['y'] = scale * self.room_exits[i][obj_idx]['y']

        # Useful for losses; for now, only rooms with no objects
        self.rooms_not_to_start_in = [i for i in range(self.n_rooms) if np.all(self.room_objects[i]['colors'] == 0.)]
        self.possible_start_rooms = [i for i in range(self.n_rooms) if not np.all(self.room_objects[i]['colors'] == 0.)]

        if self.map_name == 'DoubleDonut' and self.objects_layout_name == 'Ambiguous':
             self.possible_start_rooms = np.array([0,1,2,3,4,5,6,8,9,10,12,13,14,15]) # Do not allow start from ambiguous rooms

        logging.critical('Allowed starting rooms : {}'.format(self.possible_start_rooms))


        # # Make sure that everything is normalized reasonably
        # rooms = np.random.choice(self.possible_start_rooms, size=(1024, ))
        # xy_room = np.random.uniform(-self.scale, self.scale, size=(1024, 2))
        # images = self.get_images(rooms, xy_room)
        # logging.critical('Shape of images : {}'.format(images.shape))
        # norms = tch.sqrt(tch.mean((images.view(images.shape[0], -1)**2), dim=-1))
        # logging.critical('Norms : {}'.format(norms))
        # plt.figure()
        # norms = norms.cpu().numpy()
        # plt.hist(norms)
        # plt.show()
        #
        # plt.figure()
        # plop = images.view(images.shape[0], -1).cpu().numpy()
        # maxs = plop.max(axis=-1)
        # plt.hist(plop.flatten())
        # # plt.hist(maxs)
        # plt.show()


    def render_background(self, ax_to_use=None):
        if ax_to_use is None:
            fig, ax = plt.subplots()
        else:
            ax = ax_to_use

        ax.set_facecolor([0,0,0,0])
        ax.set_axis_off()
        ax.set_aspect('equal')

        ax.set_xlim([np.min(self.room_centers[:,0])-self.scale-.05, np.max(self.room_centers[:,0])+self.scale+.05])
        ax.set_ylim([np.min(self.room_centers[:,1])-self.scale-.05, np.max(self.room_centers[:,1])+self.scale+.05])

        # This one is common to all environments, it determines the enclosing area
        rect = patches.Rectangle((np.min(self.room_centers[:,0])-self.scale, np.min(self.room_centers[:,1]-self.scale)),
                                np.max(self.room_centers[:,0]) - np.min(self.room_centers[:,0])+ 2*self.scale,
                                np.max(self.room_centers[:,1]) - np.min(self.room_centers[:,1])+ 2*self.scale,
                                linewidth=1, edgecolor='k',facecolor=[0,0,0,0])
        ax.add_patch(rect)

        if self.map_name == 'SnakePath':
            ax.plot([-3 *self.scale, self.scale], [-self.scale, -self.scale], c='k', linewidth=3)
            ax.plot([-self.scale, 3*self.scale], [self.scale, self.scale], c='k', linewidth=3)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DonutPath':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DoubleDonut':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

            rect = patches.Rectangle((5*self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([3*self.scale, 3*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([5*self.scale, 5*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([7*self.scale, 7*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

        if ax_to_use is None:
            return fig, ax
        else:
            return ax

    def render_template(self, ax_to_use=None):
        if ax_to_use is None:
            fig, ax = self.render_background(ax_to_use=None)
        else:
            ax = self.render_background(ax_to_use=ax_to_use)

        for room in range(self.n_rooms):
            for xy, col in zip(self.room_objects[room]['positions'], self.room_objects[room]['colors']):
                xy0 = self.room_centers[room, :2]
                rect = patches.Rectangle(xy+xy0-.05*self.scale, .2*self.scale, .2*self.scale, linewidth=1, edgecolor='k', facecolor=col)
                ax.add_patch(rect)

        if ax_to_use is None:
            return fig, ax
        else:
            return ax

    def render_individual_rooms(self, ax_to_use=None):
        if __name__ != '__main__':
            raise RuntimeError("World.render_individual_rooms is meant to be called only from environment.py as a preliminary diagnostics")

        os.makedirs('out/env_render_templates/{}_{}/room_renders'.format(self.map_name, self.objects_layout_name), exist_ok=True)
        room_render_files='out/env_render_templates/{}_{}/room_renders'.format(self.map_name, self.objects_layout_name) + '/{}.pdf'

        for room in range(self.n_rooms):
            fig, ax = self.render_background(ax_to_use=None)
            for xy, col in zip(self.room_objects[room]['positions'], self.room_objects[room]['colors']):
                xy0 = self.room_centers[room, :2]
                rect = patches.Rectangle(xy+xy0-.05*self.scale, .2*self.scale, .2*self.scale, linewidth=1, edgecolor='k', facecolor=col)
                ax.add_patch(rect)

            circ = patches.Circle((self.room_centers[room,0], self.room_centers[room,1]), self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.2])
            ax.add_patch(circ)
            fig.savefig(room_render_files.format(room))
            plt.close(fig)

    def reset(self):
        # if self.map_name == 'SnakePath' and self.objects_layout_name == 'DarkCenter':
        #     self.agent_room = self.rng.choice([0,1,2,3,5,6,7,8]) # Do not allow start from dark room
        # elif self.map_name == 'DoubleDonut' and self.objects_layout_name == 'Ambiguous':
        #     self.agent_room = self.rng.choice([0,1,2,3,4,5,6,8,9,10,12,13,14,15]) # Do not allow start from ambiguous rooms
        # else:
        self.agent_room = self.rng.choice(self.possible_start_rooms)
        Lx, Ly = self.room_sizes[self.agent_room]
        self.agent_position = np.array([self.rng.uniform(-Lx, Lx), self.rng.uniform(-Ly, Ly)])
        return self.agent_room, self.agent_position

    def set_agent_position(self, room, xy=(0,0)):
        self.agent_room = room
        Lx, Ly = self.room_sizes[self.agent_room]
        invalid_x = xy[0]>Lx or xy[0]<-Lx
        invalid_y = xy[1]>Ly or xy[1]<-Ly
        if invalid_x or invalid_y:
            raise RuntimeError('Invalid xy initialization for current room')
        self.agent_position = np.array([*xy])

    # NOTE: z is now outputted directly from step
    def step(self, action):
        assert len(action) == 2
        action_bkp = deepcopy(action)
        bkp_x, bkp_y = deepcopy(self.agent_position)
        room_bkp = deepcopy(self.agent_room)

        new_pos = np.array([self.agent_position[0] + action[0], self.agent_position[1] + action[1]])
        Lx, Ly = self.room_sizes[self.agent_room]

        invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx
        invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly

        if invalid_x and invalid_y:
            if self.rng.uniform() < .5:
                action[0] -= new_pos[0] - np.clip(new_pos[0], -Lx, Lx)
                new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
                invalid_x = False
            else:
                action[1] -= new_pos[1] - np.clip(new_pos[1], -Ly, Ly)
                new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
                invalid_y = False


        changed_room = False
        if invalid_y:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='horizontal':
                    if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
                        logging.debug('crossed horizontal door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if invalid_x:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='vertical':
                    if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
                        logging.debug('crossed vertical door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if not changed_room:
            new_room = room_bkp
        else:
            new_room = used_exit['goes_to'][0]
            new_pos = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[new_room, :2]

        rectified_new_pos = np.zeros(2)
        rectified_new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
        rectified_new_pos[1] = np.clip(new_pos[1], -Ly, Ly)

        rectified_action = action + rectified_new_pos - new_pos

        self.agent_room = deepcopy(new_room)
        self.agent_position = deepcopy(rectified_new_pos)

        return new_room, rectified_new_pos, rectified_action


    def get_images(self, rooms, positions):
        # logging.critical(rooms.shape)
        # logging.critical(positions.shape)
        # logging.critical('In Legacy World : rooms {} ; positions {}'.format(rooms[0, :3], positions[0, :3]))

        desired_shape = deepcopy(rooms.shape)
        rooms = rooms.flatten().astype(int)
        positions = positions.reshape(-1, 2)
        all_positions = tch.from_numpy(self.positions_blob[rooms] - positions[:, np.newaxis, :].repeat(self.max_objects_per_room, axis=1)).float().to(self.retina.device)
        all_colors = tch.from_numpy(self.colors_blob[rooms]).float().to(self.retina.device)
        # logging.critical('In Legacy World : allpositions {} ; all_colors {}'.format(all_positions[:3], all_colors[:3]))

        image_batch = self.retina.activity(all_positions, all_colors)

        return image_batch.reshape(*desired_shape, self.retina.n, 3)


    def __replay_one_traj(self, actions, start_room=None, start_pos=None):
        n_epochs = actions.shape[0]
        positions = np.zeros((n_epochs+1,2))
        validated_actions = np.zeros((n_epochs,2))
        rooms = np.zeros((n_epochs+1))

        if start_room is None:
            room, pos = self.reset()
        else:
            room, pos = start_room, start_pos
            room = int(room)
            self.set_agent_position(room, (pos[0], pos[1]))

        positions[0] = pos
        rooms[0] = room

        for idx, action in enumerate(actions):
            new_room, new_pos, rectified_action = self.step(action)
            logging.debug('Start in room {} at ({},{}) and end in room {} at ({},{}) with tot_dep ({},{})'.format(room, *pos, new_room, *new_pos, *rectified_action))
            validated_actions[idx] = rectified_action
            positions[idx+1] = new_pos
            rooms[idx+1] = new_room
            pos = new_pos
            room = new_room

        return rooms, positions, validated_actions


    def static_replay(self, actions_batch, start_rooms=None, start_pos=None):
        actions_batch_local = deepcopy(actions_batch)
        batch_size = actions_batch.shape[0]
        epoch_len = actions_batch.shape[1]

        if start_rooms is not None:
            assert start_pos is not None
            assert start_rooms.shape[0] == start_pos.shape[0]
            assert start_pos.shape[1] == 2

        rooms = np.zeros((batch_size, epoch_len+1))
        positions = np.zeros((batch_size, epoch_len + 1, 2))
        validated_actions = np.zeros((batch_size, epoch_len, 2))

        # NOTE: making this multi-threaded seems smart, but at least in my tests its either slow, buggy, or both
        for b in range(batch_size):
            if start_rooms is None:
                room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
            else:
                room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=start_rooms[b], start_pos=start_pos[b])

            rooms[b, :] = room
            positions[b, :] = pos
            validated_actions[b, :] = act

        logging.debug('Done with static_replay in environments.py')
        return rooms, positions, validated_actions


    @staticmethod
    def __add_arrows(line, size=15, color=None, zorder=-1):

        if color is None:
            color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        x_ends = .5* (xdata[:-1] + xdata[1:])
        y_ends = .5* (ydata[:-1] + ydata[1:])

        for x_start, x_end, y_start, y_end in zip(xdata, x_ends, ydata, y_ends):
            line.axes.annotate('',
                xytext=(x_start, y_start),
                xy=(x_end, y_end),
                arrowprops=dict(arrowstyle="->", color=color),
                size=size, zorder=-1
            )



    def plot_trajectory(self, actions, start_room=None, start_pos=None, ax_to_use=None, save_file=None, color=None, marker='+', zorder=-1, show_lines=False, show_arrows=False, s=32, **kwargs):
        # By default, color is set to show time, but can be overridden
        if color is None:
            time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[0]+1)
            cmap = plt.get_cmap('jet')
            color = cmap(time_based_norm(range(actions.shape[0]+1)))

        if ax_to_use is None:
            if self.map_name != 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(5,5))
            elif self.map_name == 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(10,5))

        else:
            ax = ax_to_use

        if start_room is None:
            room, pos, act =  self.__replay_one_traj(actions, start_room=None, start_pos=None)
        else:
            room, pos, act =  self.__replay_one_traj(actions, start_room=start_room, start_pos=start_pos)

        ax = self.render_template(ax_to_use=ax)
        absolute_pos = pos+self.room_centers[room.astype(int), :2]

        ax.scatter(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, marker=marker, s=s)

        if show_lines:
            lines = ax.plot(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, ls='-')
            if show_arrows:
                self.__add_arrows(lines[0], zorder=zorder, color=color)

        if save_file is not None:
            os.makedirs('/'.join(save_file.split('/')[:-1]), exist_ok=True)
            fig.savefig(save_file)
            plt.close(fig)
        else:
            return ax











































if __name__ == '__main__':
    os.makedirs('out/env_render_templates/', exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    map_name = 'SnakePath'
    objects_layout_name = 'Default'

    # env = LegacyWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos='plo')
    # env = FixedRewardWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos='Default')
    #
    # raise RuntimeError

    for chosen_reward_pos in ['None', 'Default', 'TopRight']:
        env = FixedRewardWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos=chosen_reward_pos)
        # env = World(map_name=map_name, objects_layout_name=objects_layout_name)
        fig, ax = env.render_template()
        fig.savefig('out/env_render_templates/{}_{}_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos))
        plt.close('all')

        actions_batch = env.scale * np.random.randn(10, 20, 2)
        start_rooms = np.random.choice(env.n_rooms, size=(100,))
        start_pos = np.random.uniform(-env.scale, env.scale, size=(100, 2))
        bkp_actions = deepcopy(actions_batch)
        _, _, validated_actions = env.static_replay(actions_batch, start_rooms=start_rooms, start_pos=start_pos)

        assert np.all(bkp_actions == actions_batch), "Side-effect detected in static replay"
        assert not np.all(validated_actions == bkp_actions), "Static replay made no change to any of the trajectories, this is suspicious"

        for b in range(10):
            env.plot_trajectory(validated_actions[b], start_room=start_rooms[b], start_pos=start_pos[b], save_file='out/env_render_templates/{}_{}_{}/trajectories/traj_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos, b))

        if map_name in ['DoubleDonut', 'SnakePath']:
            deliberate_actions = np.reshape(meaningful_trajectories[map_name] * env.scale, (1,)+meaningful_trajectories[map_name].shape)
            print(deliberate_actions)
            rooms, positions, deliberate_actions = env.static_replay(deliberate_actions, start_rooms=np.zeros(5, dtype=int), start_pos=np.zeros((5, 2)))
            print(deliberate_actions)

            deliberate_actions = deliberate_actions[0]
            rooms = rooms[0].astype(int)
            positions = positions[0]

            fig, ax = plt.subplots(figsize=(10,5))
            ax = env.plot_trajectory(deliberate_actions, start_room=0, start_pos=np.array([0.,0.]), ax_to_use=ax, marker = 'x', zorder=-5)
            fig.savefig('out/env_render_templates/{}_{}_{}/trajectories/deliberate_traj.pdf'.format(map_name, objects_layout_name, chosen_reward_pos))

            for b in range(5):
                perturbed_positions = np.clip(positions+.2*env.scale*np.random.uniform(-1, 1, size=positions.shape), -env.scale, env.scale)
                global_positions = perturbed_positions + env.room_centers[rooms, :2]
                perturbed_actions = global_positions[1:] - global_positions[:-1]

                fig, ax = plt.subplots(figsize=(10,5))
                # env.plot_trajectory(deliberate_actions + .02*env.scale*np.mean(np.abs(deliberate_actions))*np.random.randn(*deliberate_actions.shape), ax_to_use=ax, start_room=0, start_pos=np.array([0.,0.]))
                env.plot_trajectory(perturbed_actions, ax_to_use=ax, start_room=0, start_pos=np.array([0.,0.]))
                fig.savefig('out/env_render_templates/{}_{}_{}/trajectories/deliberate_traj_noisy_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos, b))
