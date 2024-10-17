# EvaluateRF.py

import joblib
import numpy as np

rf = joblib.load('/afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7/Machile_Learning/Models/random_forest_parametric_model_RFC_.pkl')

def load_random_forest_model(inputs):
    try:
        result = rf.predict_proba(np.array(inputs).reshape(1, -1))
        return [result[0][1]]
    except Exception as e:
        return [-99.9]

if __name__ == "__main__":
    # You can include some test code here for local testing
    test_input = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    second_test = [145.73484802246094, 120.59391021728516, -0.443359375, -0.7113037109375, 37.01284408569336, 552.005859375, 408.7455749511719, 371.141357421875, 266.1398010253906, 0.2784280478954315, 0.07568359375, 286.85198974609375, 0.2679443359375, 281.60650634765625, 33.007198333740234, 278.4770202636719, 317.86181640625, 617.2410278320312, 576.8833618164062, 96.34343719482422, 131.87417602539062, 3.0071299076080322, 3.0485455989837646, 42.356231689453125, 167.6221160888672, 0.0, 0.25672540068626404, 0.023939067497849464, 1.4265251159667969, 2.8630480766296387, -2.0, -2.0, 160.0, 150.0, 800.0]
    result = load_random_forest_model(second_test)
    print("Result:", result)
