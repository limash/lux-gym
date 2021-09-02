import numpy as np

actions_number = 40
eye = np.eye(actions_number, dtype=np.half)

citytile_action_mask = np.zeros(actions_number, dtype=np.half)
citytile_action_mask[:4] = 1
worker_action_mask = np.zeros(actions_number, dtype=np.half)
worker_action_mask[4:23] = 1
cart_action_mask = np.zeros(actions_number, dtype=np.half)
cart_action_mask[23:] = 1

action_vector = {
    "ct_build_worker": eye[0],
    "ct_build_cart": eye[1],
    "ct_research": eye[2],
    "ct_idle": eye[3],
    "w_mn": eye[4],
    "w_me": eye[5],
    "w_ms": eye[6],
    "w_mw": eye[7],
    "w_mc": eye[8],
    "w_pillage": eye[9],
    "w_tnwood": eye[10],
    "w_tewood": eye[11],
    "w_tswood": eye[12],
    "w_twwood": eye[13],
    "w_tncoal": eye[14],
    "w_tecoal": eye[15],
    "w_tscoal": eye[16],
    "w_twcoal": eye[17],
    "w_tnuranium": eye[18],
    "w_teuranium": eye[19],
    "w_tsuranium": eye[20],
    "w_twuranium": eye[21],
    "w_build": eye[22],
    "c_mn": eye[23],
    "c_me": eye[24],
    "c_ms": eye[25],
    "c_mw": eye[26],
    "c_mc": eye[27],
    "c_tnwood": eye[28],
    "c_tewood": eye[29],
    "c_tswood": eye[30],
    "c_twwood": eye[31],
    "c_tncoal": eye[32],
    "c_tecoal": eye[33],
    "c_tscoal": eye[34],
    "c_twcoal": eye[35],
    "c_tnuranium": eye[36],
    "c_teuranium": eye[37],
    "c_tsuranium": eye[38],
    "c_twuranium": eye[39],
}
