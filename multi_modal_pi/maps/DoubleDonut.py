import numpy as np

doubledonut_layout_dict = {
'room_centers':
    np.array([
        # 1st donut (left)
        [-2., -2., 0],
        [0., -2., 0],
        [2., -2., 0],
        [2., 0., 0],
        [2., 2., 0],
        [0., 2., 0],
        [-2., 2., 0],
        [-2., 0., 0],

        # 2nd donut (right)
        [4., -2., 0],
        [6., -2., 0],
        [8., -2., 0],
        [8., 0., 0],
        [8., 2., 0],
        [6., 2., 0],
        [4., 2., 0],
        [4., 0., 0],

    ]),

'room_sizes':
    np.array([
        # 1st floor
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],

        # 2nd floor
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
    [ # Have to add exits only to 2, 3, 4
        [{'goes_to': [1, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},
         {'goes_to': [7, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.}, ], # Room 0 exits
        [{'goes_to': [2, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
          {'goes_to': [0, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 1 exits
        [{'goes_to': [3, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
          {'goes_to': [1, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
          {'goes_to': [8, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 2 exits
        [{'goes_to': [4, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
          {'goes_to': [2, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
          {'goes_to': [15, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 3 exits
        [{'goes_to': [5, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y': 0.},
          {'goes_to': [3, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
          {'goes_to': [14, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 4 exits
        [{'goes_to': [6, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
           {'goes_to': [4, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 5 exits
        [{'goes_to': [7, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
          {'goes_to': [5, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},], # Room 6 exits
        [{'goes_to': [6, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
           {'goes_to': [0, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 7 exits

          # Add only to 8, 14, 15
        [{'goes_to': [9, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
        {'goes_to': [15, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y':1.},
        {'goes_to': [2, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 8 exits
        [{'goes_to': [10, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},
          {'goes_to': [8, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 9 exits
        [{'goes_to': [11, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
          {'goes_to': [9, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 10 exits
        [{'goes_to': [12, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
          {'goes_to': [10, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 11 exits
        [{'goes_to': [13, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y': 0.},
          {'goes_to': [11, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 12 exits
        [{'goes_to': [14, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
           {'goes_to': [12, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y':0.},], # Room 13 exits
        [{'goes_to': [15, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},
          {'goes_to': [13, [-1., 0.]], 'axis': 'vertical', 'x': 1., 'y': 0.},
          {'goes_to': [4, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},], # Room 14 exits
        [{'goes_to': [14, [0., -1.]], 'axis': 'horizontal', 'x': 0., 'y': 1.},
        {'goes_to': [3, [1., 0.]], 'axis': 'vertical', 'x': -1., 'y':0.},
        {'goes_to': [8, [0., 1.]], 'axis': 'horizontal', 'x': 0., 'y': -1.},], # Room 15 exits
    ],
'possible_reward_pos': {
    'Default': {'room': 4, 'pos': [0., 0.]},
},


'possible_layouts':
# For right donut, keep the same corner colors but double the objects (y =  +- .5)

    {'Default':
        [ # 'positions': (n_obj, 2) colors(n_obj, 3)
            {'positions': np.array([
                [0., 0.],
                [2., 0.],
                [0., 2.],
                [4., 0.],
                [0., 4.],
                         ]),
             'colors': np.array([
                [1., 0., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 1., 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
             ])
            }, # Room 0

            {'positions': np.array([
                [0., 0.],
                [-2., 0.],
                [2., 0.],
                [4., .5],
                [4., -.5],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1., 0., 0.],
                [0., 1., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
             ])
            }, # Room 1

            {'positions': np.array([
                [0., 0.],
                [-2., 0.],
                [0., 2.],
                [-4., 0.],
                [0., 4.],
                [2., .5],
                [2., -.5],
                [4., .5],
                [4., -.5],
                [2., 1.5],
                [2., 2.5],
                         ]),
             'colors': np.array([
                [0., 1., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1., 0., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
             ])
             }, # Room 2

            {'positions': np.array([
                [0., 0.],
                [0., 2.],
                [0., -2.],
                [2., .5],
                [2., -.5],
                [2., 1.5],
                [2., 2.5],
                [2., -1.5],
                [2., -2.5],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [0., 1., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
             ])
            }, # Room 3

            {'positions': np.array([
                [0., 0.],
                [-2., 0.],
                [0., -2.],
                [-4., 0.],
                [0., -4.],
                [2., .5],
                [2., -.5],
                [4., .5],
                [4., -.5],
                [2., -1.5],
                [2., -2.5],
                         ]),
             'colors': np.array([
                [0., 0., 1.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [0., 1., 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
             ])
             }, # Room 4

            {'positions': np.array([
                [0., 0.],
                [2., 0.],
                [-2., 0.],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
             ])
            }, # Room 5

            {'positions': np.array([
                [0., 0.],
                [2., 0.],
                [0., -2.],
                [4., 0.],
                [0., -4.],
                         ]),
             'colors': np.array([
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [1., 0., 1.],
             ])
             }, # Room 6

            {'positions': np.array([
                [0., 0.],
                [0., 2.],
                [0., -2],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1., 0., 0.],
             ])
            }, # Room 7

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [2., 0.5],
                [2., -0.5],
                [4, 0.5],
                [4, -0.5],
                [0., 1.5],
                [0., 2.5],
                [0., 3.5],
                [0., 4.5],
                [-2, 0.],
                [-4, 0.],
                [-2., 2.]
                         ]),
             'colors': np.array([
                [1., 0., 0.],
                [1., 0., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 1., 0.],
                [0., 1., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [0., 1., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
             ])
            }, # Room 8

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [-2., -0.5],
                [-2., 0.5],
                [2., 0.5],
                [2., -0.5],
                [-4., 0.]
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
             ])
            }, # Room 9

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [-2., 0.5],
                [-2., -0.5],
                [0., 2.5],
                [0., 1.5],
                [-4., 0.5],
                [-4., -0.5],
                [0., 4.5],
                [0., 3.5],
                         ]),
             'colors': np.array([
                [0., 1., 0.],
                [0., 1., 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 0., 1.],
                [0., 0., 1.],
             ])
             }, # Room 10

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [0., 2.5],
                [0., 1.5],
                [0., -1.5],
                [0., -2.5],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [0., 0., 1.],
                [0., 1., 0.],
                [0., 1., 0.],
             ])
            }, # Room 11

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [-2., 0.5],
                [-2., -0.5],
                [0., -2.5],
                [0., -1.5],
                [-4., 0.5],
                [-4., -0.5],
                [0., -3.5],
                [0., -4.5],
                         ]),
             'colors': np.array([
                [0., 0., 1.],
                [0., 0., 1.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [0., 1., 0.],
                [0., 1., 0.],
             ])
             }, # Room 12

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [2., 0.5],
                [2., -0.5],
                [-2., 0.5],
                [-2., -0.5],
                [-4., 0.],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [0., 0., 1.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [0., 0., 1.],
             ])
            }, # Room 13

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [2., 0.5],
                [2., -0.5],
                [0., -2.5],
                [0., -1.5],
                [4., 0.5],
                [4., -0.5],
                [-2., 0.],
                [-4., 0.],
                [-2., -2.],
                [0., -3.5],
                [0., -4.5],
                         ]),
             'colors': np.array([
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 0., 1.],
                [0., 0., 1.],
                [0., 0., 1.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1., 0., 0.],
                [1., 0., 0.],
             ])
             }, # Room 14

            {'positions': np.array([
                [0., 0.5],
                [0., -0.5],
                [0., 1.5],
                [0., 2.5],
                [0., -1.5],
                [0., -2.5],
                [-2., 2.],
                [-2., 0],
                [-2., -2],
                         ]),
             'colors': np.array([
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 0., 1.],
                [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                [0., 1., 0.],
             ])
            }, # Room 15

        ],

    # Make the two opposite ends of the "infinite" identical, so that 7 and 11 are ambiguous, force no resetting
    'Ambiguous': [ # 'positions': (n_obj, 2) colors (n_obj, 3)
        {'positions': np.array([
            [0., 0.],
            [2., 0.],
            [0., 2.],
            [4., 0.],
            [0., 4.],
                     ]),
         'colors': np.array([
            [1., 0., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [0., 1., 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
         ])
        }, # Room 0

        {'positions': np.array([
            [0., 0.],
            [-2., 0.],
            [2., 0.],
            [4., .5],
            [4., -.5],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
         ])
        }, # Room 1

        {'positions': np.array([
            [0., 0.],
            [-2., 0.],
            [0., 2.],
            [-4., 0.],
            [0., 4.],
            [2., .5],
            [2., -.5],
            [4., .5],
            [4., -.5],
            [2., 1.5],
            [2., 2.5],
                     ]),
         'colors': np.array([
            [0., 1., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
         ])
         }, # Room 2

        {'positions': np.array([
            [0., 0.],
            [0., 2.],
            [0., -2.],
            [2., .5],
            [2., -.5],
            [2., 1.5],
            [2., 2.5],
            [2., -1.5],
            [2., -2.5],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [0., 0., 1.],
            [0., 1., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
         ])
        }, # Room 3

        {'positions': np.array([
            [0., 0.],
            [-2., 0.],
            [0., -2.],
            [-4., 0.],
            [0., -4.],
            [2., .5],
            [2., -.5],
            [4., .5],
            [4., -.5],
            [2., -1.5],
            [2., -2.5],
                     ]),
         'colors': np.array([
            [0., 0., 1.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [0., 1., 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
         ])
         }, # Room 4

        {'positions': np.array([
            [0., 0.],
            [2., 0.],
            [-2., 0.],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [0., 0., 1.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
         ])
        }, # Room 5

        {'positions': np.array([
            [0., 0.],
            [2., 0.],
            [0., -2.],
            [4., 0.],
            [0., -4.],
                     ]),
         'colors': np.array([
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [0., 0., 1.],
            [1., 0., 1.],
         ])
         }, # Room 6

        {'positions': np.array([
            [0., 0.],
            [0., 2.],
            [0., -2],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1., 0., 0.],
         ])
        }, # Room 7

        {'positions': np.array([
            [0., 0.5],
            [0., -0.5],
            [2., 0.5],
            [2., -0.5],
            [4, 0.],
            [0., 1.5],
            [0., 2.5],
            [0., 3.5],
            [0., 4.5],
            [-2, 0.],
            [-4, 0.],
            [-2., 2.]
                     ]),
         'colors': np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [0., 1., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
         ])
        }, # Room 8

        {'positions': np.array([
            [0., 0.5],
            [0., -0.5],
            [-2., -0.5],
            [-2., 0.5],
            [2., 0.],
            [-4., 0.]
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
         ])
        }, # Room 9

        {'positions': np.array([
            [0., 0.],
            [-2., 0.5],
            [-2., -0.5],
            [0., 2],
            [-4., 0.5],
            [-4., -0.5],
            [0., 4.],

                     ]),
         'colors': np.array([
            [1., 0., 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [1., 0., 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
         ])
         }, # Room 10

        {'positions': np.array([
            [0., 0.],
            [0., 2.],
            [0., -2.],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1., 0., 0.],

         ])
        }, # Room 11

        {'positions': np.array([
            [0., 0.],
            [-2., 0.5],
            [-2., -0.5],
            [0., -2.],
            [-4., 0.5],
            [-4., -0.5],
            [0., -4.],
                     ]),
         'colors': np.array([
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1., 0., 0.],
         ])
         }, # Room 12

        {'positions': np.array([
            [0., 0.5],
            [0., -0.5],
            [2., 0.],
            [-2., 0.5],
            [-2., -0.5],
            [-4., 0.],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [0., 0., 1.],
         ])
        }, # Room 13

        {'positions': np.array([
            [0., 0.5],
            [0., -0.5],
            [2., 0.5],
            [2., -0.5],
            [0., -2.5],
            [0., -1.5],
            [4., 0.],
            [-2., 0.],
            [-4., 0.],
            [-2., -2.],
            [0., -3.5],
            [0., -4.5],
                     ]),
         'colors': np.array([
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [0., 0., 1.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1., 0., 0.],
            [1., 0., 0.],
         ])
         }, # Room 14

        {'positions': np.array([
            [0., 0.5],
            [0., -0.5],
            [0., 1.5],
            [0., 2.5],
            [0., -1.5],
            [0., -2.5],
            [-2., 2.],
            [-2., 0],
            [-2., -2],
                     ]),
         'colors': np.array([
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1./np.sqrt(2), 1./np.sqrt(2), 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
            [0., 1., 0.],
         ])
        }, # Room 15

    ],
    },
}

doubledonut_trajectory = np.array([
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
