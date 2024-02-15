"""This file contains the specifications of the attributes in the dataset."""

import pandas as pd

ATTRIBUTES = {
    0: 'seconds',
    1: 'external_arousal',
    2: 'external_valence',
    3: 'partner_arousal',
    4: 'partner_valence',
    5: 'self_arousal',
    6: 'self_valence',
    7: 'x',
    8: 'y',
    9: 'z',
    10: 'E4_BVP',
    11: 'E4_EDA',
    12: 'E4_HR',
    13: 'E4_IBI',
    14: 'E4_TEMP',
    15: 'Attention',
    16: 'delta',
    17: 'lowAlpha',
    18: 'highAlpha',
    19: 'lowBeta',
    20: 'highBeta',
    21: 'lowGamma',
    22: 'middleGamma',
    23: 'theta',
    24: 'Meditation'
}

CLASS_LABELS = pd.Series(range(6, 7))

DATA_LABELS = pd.Series(range(7, 25))

SKEWED_DATA = pd.Series([10, 13, 16, 17, 18, 19, 20, 21, 22, 23])

BRAINWAVE_BANDS = pd.Series(range(16, 24))

CATEGORICAL_LABELS = pd.Series([6, 15, 24])
