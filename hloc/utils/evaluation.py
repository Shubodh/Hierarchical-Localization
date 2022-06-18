import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import sys

from typing import Tuple, List, Dict
from tabulate import tabulate

def quaternion_inverse(given_quat):
    """
    Input: scipy format - scalar LAST. numpy.array([b, c, d, a]) where quaternion is a + bi + cj + dk
    or in wxyz format, w is scalar below
    Output: numpy array in same format but inverse of input
    """
    #pyquaternion library: Scalar FIRST: numpy.array([a, b, c, d]) where quaternion is a + bi + cj + dk
    qx, qy, qz, qw = given_quat #scalar last
    given_quat_scalar_first = np.array([qw, qx, qy, qz]) #scalar first
    quat_obj = Quaternion(array=given_quat_scalar_first) 
    inv_quat = quat_obj.inverse
    inv_quat_np = inv_quat.elements

    iqw, iqx, iqy, iqz  = inv_quat_np #scalar first
    inv_quat_np_scalar_first = np.array([iqx, iqy, iqz, iqw]) #scalar  last
    return  inv_quat_np_scalar_first


def get_rotation_error_using_arccos_of_trace_of_rotation(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
        # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    # Returns angle in degrees
    if R_pred.shape == (3,3):
        R_pred = R_pred[np.newaxis,:]
    if R_gt.shape == (3,3):
        R_gt = R_gt[np.newaxis,:]
    n = R_gt.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_pred, R_gt.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric_in_degrees = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric_in_degrees[0]


def get_rotation_error_using_quaternion_dot(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
    # cpp
    # const float d1 = std::fabs(q1.dot(q2));
    #d2 = std::fmin(1.0f, std::fmax(-1.0f, d1))
    #return 2 * acos(d2) * 180 / M_PI

    # return np.linalg.norm(logm(np.matmul(R_pred, R_gt.transpose()))) / np.sqrt(2)
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError(f'Input matrices must be 3x3, instead got {R_pred.shape} and {R_gt.shape}')

    R_pred_obj, R_gt_obj = R.from_matrix(R_pred), R.from_matrix(R_gt)
    quat_pred, quat_gt = R_pred_obj.as_quat(), R_gt_obj.as_quat()

    # quat_pred_inv, quat_gt_inv = quaternion_inverse(quat_pred), quaternion_inverse(quat_gt)
    # quat_pred_inv_i, quat_gt_inv_i = quaternion_inverse(quat_pred_inv), quaternion_inverse(quat_gt_inv)
    # print(quat_pred, quat_gt)
    # print(quat_pred_inv, quat_gt_inv)
    # print(quat_pred_inv_i, quat_gt_inv_i)

    # d1 = abs(np.dot(quat_pred, quat_gt))
    # print("Quat inverse")
    d1 = abs(np.dot(quaternion_inverse(quat_pred), quat_gt))
    d2 = np.min((1.0, np.max((-1.0, d1))))
    rotation_error_in_degrees = 2 * np.arccos(d2) * 180 / np.pi
    # print(rotation_error_in_degrees)
    return rotation_error_in_degrees

def get_rotation_error_using_log_of_rotation(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError(f'Input matrices must be 3x3, instead got {R_pred.shape} and {R_gt.shape}')

    return np.linalg.norm(logm(np.matmul(R_pred, R_gt.transpose()))) / np.sqrt(2)

def get_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    if t_gt.shape != (3, ) or t_gt.shape != (3, ):
        raise ValueError(f'Input vectors must be 3 dimensional, instead got {t_pred.shape} and {t_gt.shape}')

    return np.linalg.norm(t_gt - t_pred)

def get_both_errors(T_pred: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    if T_pred.shape != (4, 4) or T_gt.shape != (4, 4):
        raise ValueError(f'Input matrices must be 4x4, instead got {T_pred.shape} and {T_gt.shape}')

    # rot_error = get_rotation_error_using_quaternion_dot(T_pred[:3, :3], T_gt[:3, :3])
    # print(rot_error, "quat")
    rot_error = get_rotation_error_using_arccos_of_trace_of_rotation(T_pred[:3, :3], T_gt[:3, :3])
    # print(rot_error, "trace")
    trans_error = get_translation_error(T_pred[:3, -1], T_gt[:3, -1])
    return (rot_error, trans_error)

def get_errors(preds: List[np.ndarray], truths: List[np.ndarray]) -> Tuple[List[float], List[float]]:
    if len(preds) != len(truths):
        raise ValueError(f'Input must consist of equal numbers of predictions and ground truths, instead got {len(preds)} and {len(truths)}')
    
    rot_errors = []
    trans_errors = []

    for T_pred, T_gt in zip(preds, truths):
        rot_err, trans_err = get_both_errors(T_pred, T_gt)
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)

    return rot_errors, trans_errors

# can add similar functions for other file formats
def read_3DMatch_file(path: str) -> List[np.ndarray]:
    with open(path, 'r') as f:
        lines = filter(lambda x: len(x.split()) == 4, f)
        long_arr = np.genfromtxt(lines)
        return np.split(long_arr, long_arr.shape[0]/4)

def print_statistics(data: List[Dict]) -> None:
    table = []
    for dataset in data:
        res = {
            'Dataset': dataset['name'],
            'Rot. Error': np.mean(dataset['rot_errors']),
            'Trans. Error': np.mean(dataset['trans_errors']),
        }
        table.append(res)

    # print results
    print(tabulate(table, headers='keys', tablefmt='presto'))

    # histogram
    fig, axs = plt.subplots(len(data), 2)
    axs = axs.reshape((len(data), 2)) # when len(data) = 1
    for i in range(len(data)):
        r = data[i]['rot_errors']
        t = data[i]['trans_errors']
        
        axs[i, 0].hist(r, weights=np.ones(len(r))/len(r))
        axs[i, 0].set_title(data[i]['name'])
        axs[i, 0].set(xlabel='Rot. Error', ylabel='Fraction')

        axs[i, 1].hist(t, weights=np.ones(len(t))/len(t))
        axs[i, 1].set_title(data[i]['name'])
        axs[i, 1].set(xlabel='Trans. Error', ylabel='Fraction')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data = []

    # 3DMatch
    preds = read_3DMatch_file('data.pred')
    truths = read_3DMatch_file('data.gt')
    rot_errors, trans_errors = get_errors(preds, truths)
    data.append({
        'name': 'GT + Random Noise',
        'rot_errors': rot_errors,
        'trans_errors': trans_errors,
    })

    # read from other datasets/files here

    print_statistics(data)


