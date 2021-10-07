import numpy as np

snakepath_layout_dict = {
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
    'None': {'room': None, 'pos': [0., 0.]},
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
                [0., -2.],
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
}


snakepath_trajectory = np.array([
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
            ])
