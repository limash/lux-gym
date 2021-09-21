import numpy as np

actions_number = 6
eye = np.eye(actions_number, dtype=np.half)

# citytile_action_mask = np.zeros(actions_number, dtype=np.half)
# citytile_action_mask[:4] = 1
# worker_action_mask = np.zeros(actions_number, dtype=np.half)
# worker_action_mask[4:23] = 1
# cart_action_mask = np.zeros(actions_number, dtype=np.half)
# cart_action_mask[23:] = 1
#
action_vector = {
    "w_mn": eye[0],
    "w_me": eye[1],
    "w_ms": eye[2],
    "w_mw": eye[3],
    "w_mc": eye[4],
    "w_build": eye[5],
    "c_mn": eye[0],
    "c_me": eye[1],
    "c_ms": eye[2],
    "c_mw": eye[3],
    "c_mc": eye[4],
}
meaning_vector = {
    0: ("m", "n"),
    1: ("m", "e"),
    2: ("m", "s"),
    3: ("m", "w"),
    4: ("m", "c"),
    5: ("bcity",),
}

actions_number_ct = 4
eye_ct = np.eye(actions_number_ct, dtype=np.half)
action_vector_ct = {
    "ct_build_worker": eye_ct[0],
    "ct_build_cart": eye_ct[1],
    "ct_research": eye_ct[2],
    "ct_idle": eye_ct[3],
}
meaning_vector_ct = {
    0: ("bw",),
    1: ("bc",),
    2: ("r",),
    3: ("idle",),
}
# action_vector = {
#     "ct_build_worker": eye[0],
#     "ct_build_cart": eye[1],
#     "ct_research": eye[2],
#     "ct_idle": eye[3],
#     "w_mn": eye[4],
#     "w_me": eye[5],
#     "w_ms": eye[6],
#     "w_mw": eye[7],
#     "w_mc": eye[8],
#     "w_pillage": eye[9],
#     "w_tnwood": eye[10],
#     "w_tewood": eye[11],
#     "w_tswood": eye[12],
#     "w_twwood": eye[13],
#     "w_tncoal": eye[14],
#     "w_tecoal": eye[15],
#     "w_tscoal": eye[16],
#     "w_twcoal": eye[17],
#     "w_tnuranium": eye[18],
#     "w_teuranium": eye[19],
#     "w_tsuranium": eye[20],
#     "w_twuranium": eye[21],
#     "w_build": eye[22],
#     "c_mn": eye[23],
#     "c_me": eye[24],
#     "c_ms": eye[25],
#     "c_mw": eye[26],
#     "c_mc": eye[27],
#     "c_tnwood": eye[28],
#     "c_tewood": eye[29],
#     "c_tswood": eye[30],
#     "c_twwood": eye[31],
#     "c_tncoal": eye[32],
#     "c_tecoal": eye[33],
#     "c_tscoal": eye[34],
#     "c_twcoal": eye[35],
#     "c_tnuranium": eye[36],
#     "c_teuranium": eye[37],
#     "c_tsuranium": eye[38],
#     "c_twuranium": eye[39],
# }
#
# meaning_vector = {
#     0: ("bw",),
#     1: ("bc",),
#     2: ("r",),
#     3: ("idle",),
#     4: ("m", "n"),
#     5: ("m", "e"),
#     6: ("m", "s"),
#     7: ("m", "w"),
#     8: ("m", "c"),
#     9: ("p",),
#     10: ("t", "n", "wood"),
#     11: ("t", "e", "wood"),
#     12: ("t", "s", "wood"),
#     13: ("t", "w", "wood"),
#     14: ("t", "n", "coal"),
#     15: ("t", "e", "coal"),
#     16: ("t", "s", "coal"),
#     17: ("t", "w", "coal"),
#     18: ("t", "n", "uranium"),
#     19: ("t", "e", "uranium"),
#     20: ("t", "s", "uranium"),
#     21: ("t", "w", "uranium"),
#     22: ("bcity",),
#     23: ("m", "n"),
#     24: ("m", "e"),
#     25: ("m", "s"),
#     26: ("m", "w"),
#     27: ("m", "c"),
#     28: ("t", "n", "wood"),
#     29: ("t", "e", "wood"),
#     30: ("t", "s", "wood"),
#     31: ("t", "w", "wood"),
#     32: ("t", "n", "coal"),
#     33: ("t", "e", "coal"),
#     34: ("t", "s", "coal"),
#     35: ("t", "w", "coal"),
#     36: ("t", "n", "uranium"),
#     37: ("t", "e", "uranium"),
#     38: ("t", "s", "uranium"),
#     39: ("t", "w", "uranium"),
# }
