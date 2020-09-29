import numpy as np
import matplotlib.pyplot as plt

def precision_recall(gt_soft, gt_hard, S, plot_PR=True):
    '''Calculates precision, recall and F1-score for different thresholds'''
    # remove gt_soft, but keep gt_hard entries
    S[np.logical_and(gt_soft, np.logical_not(gt_hard))] = np.amin(S)

    R=[0]
    P=[1]
    startV = np.amax(S)
    endV = np.amin(S)
    bestF = 0
    bestT = startV
    bestP = 0
    bestR = 0

    # gt_sparse = sparse.csr_matrix(gt_hard) # exploit (mostly) sparsity of gt-matrix
    for i in np.linspace(startV, endV, 100):
        B = np.greater_equal(S, i)

        TP = np.count_nonzero(np.logical_and(gt_hard, B))
        FN = np.count_nonzero(np.logical_and(gt_hard, np.logical_not(B)))
        FP = np.count_nonzero(np.logical_and(np.logical_not(gt_hard), B))

        P.append(TP/(TP + FP))
        R.append(TP/(TP + FN))

        if P[-1]==0.0 and R[-1]==0.0:
            F = 0.0
        else:
            F = 2 * P[-1] * R[-1] / (P[-1] + R[-1])

        if F > bestF:
            bestF = F
            bestT = i
            bestP = P[-1]
            bestR = R[-1]

    R.append(1)
    P.append(0)
    R = np.asarray(R)
    P = np.asarray(P)
    AP = np.sum((R[1:] - R[:-1]) * P[1:])

    print("Best threshold: {:.4f}".format(bestT))
    print("Best F: {:.4f}".format(bestF))
    print("Best P: {:.4f}".format(bestP))
    print("Best R: {:.4f}".format(bestR))
    print("AP score: {:.4f}".format(AP))

    if plot_PR:
        plt.plot(R, P)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Kurve: AP={0:0.2f}'.format(AP))
        plt.show()

    return bestF, bestP, bestR, AP, R, P
