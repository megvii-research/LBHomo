import numpy as np
import cv2


def compute_error_forward(flow, pt_pairs):
    err = 0
    num_pt = len(pt_pairs)
    # num_pt = len(pt_pairs)
    for j in range(num_pt):
        xa, ya = pt_pairs[j][0]
        xa, ya = int(xa), int(ya)
        xb, yb = pt_pairs[j][1]
        dx, dy = flow[ya, xa]
        xbp = xa + dx
        ybp = ya + dy
        err += np.sqrt((xb-xbp)**2+(yb-ybp)**2)
        # print(xa, ya, xb, yb, xbp, ybp)
        # exit()
    return err/num_pt


def compute_error_backward(flow, pt_pairs):
    err = 0
    num_pt = len(pt_pairs)
    # num_pt = len(pt_pairs)
    for j in range(num_pt):
        xa, ya = pt_pairs[j][0]
        xb, yb = pt_pairs[j][1]
        xb, yb = int(xb), int(yb)
        dx, dy = flow[yb, xb]
        xbp = xa - dx
        ybp = ya - dy
        err += np.sqrt((xb-xbp)**2+(yb-ybp)**2)
        # print(xa, ya, xb, yb, xbp, ybp)
        # exit()
    return err/num_pt


def identity_error(pt_pairs, dist_pairs):
    H = np.identity(3)
    err = 0
    num_pt = len(pt_pairs)
    for j in range(num_pt):
        pt_pairs_warp = cv2.perspectiveTransform(pt_pairs[j].reshape(1, -1, 2), H).reshape(2)
        xa, ya = pt_pairs_warp
        xb, yb = dist_pairs[j]
        err += np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    return err / num_pt


def compute_error(flow_f, flow_b, pt_pairs):

    return min(compute_error_forward(flow_f, pt_pairs), compute_error_backward(flow_b, pt_pairs))