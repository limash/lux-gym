import numpy as np

# [general, direction, res_type]
empty_worker_action_vectors = [np.zeros(4, dtype=np.half), np.zeros(4, dtype=np.half), np.zeros(3, dtype=np.half)]
empty_cart_action_vectors = [np.zeros(3, dtype=np.half), np.zeros(4, dtype=np.half), np.zeros(3, dtype=np.half)]

worker_actions_number = 4
worker_eye = np.eye(worker_actions_number, dtype=np.half)
worker_action_vector = {
    "m": worker_eye[0],
    "t": worker_eye[1],
    "idle": worker_eye[2],
    "bcity": worker_eye[3],
}

cart_actions_number = 3
cart_eye = np.eye(cart_actions_number, dtype=np.half)
cart_action_vector = {
    "m": cart_eye[0],
    "t": cart_eye[1],
    "idle": cart_eye[2],
}

dir_actions_number = 4
dir_eye = np.eye(dir_actions_number, dtype=np.half)
dir_action_vector = {
    "n": dir_eye[0],
    "e": dir_eye[1],
    "s": dir_eye[2],
    "w": dir_eye[3],
}

res_actions_number = 3
res_eye = np.eye(res_actions_number, dtype=np.half)
res_action_vector = {
    "wood": res_eye[0],
    "coal": res_eye[1],
    "uranium": res_eye[2],
}

actions_number_ct = 4
eye_ct = np.eye(actions_number_ct, dtype=np.half)
action_vector_ct = {
    "ct_build_worker": eye_ct[0],
    "ct_build_cart": eye_ct[1],
    "ct_research": eye_ct[2],
    "ct_idle": eye_ct[3],
}
