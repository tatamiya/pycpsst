import numpy as np


class ChangePointSST():
    '''
    Change point detection based on singular spectrum transform.

    Parameters
    ========
    L_train: int (default 150)
        Data length of the reference term
        from which to create a train trajectory matirx.

    L_target: int (default None)
        Data length of the target term.
        from which to create a target trajectory matirx.
        If None, it is set the same as L_train.

    w: int (default 20)
        The length of a sliding window.

    d_train: int (default 3)
        The number of the left singular vectors to select
        from the train trajectory matrix.

    d_test: int (default 2)
        The number of the left singular vectors to select
        from the target trajectory matrix
    '''

    def __init__(self, L_train=150, L_target=None, w=20,
                 d_train=3, d_target=2):

        self.L_train = L_train
        if L_target is None:
            self.L_target = L_train
        else:
            self.L_target = L_target

        self.w = w
        self.r, self.m = d_train, d_target

    def _stack_svd(self, X):

        # Stack sliding windows into a matrix.
        K = X.shape[0] - self.w + 1  # The number of sliding windows to stack
        Z = np.column_stack([X[i:i + self.w] for i in range(0, K)])

        # Left singular matrix
        U, s, V = np.linalg.svd(Z)

        return U, s, V

    def score(self, X):

        window_train = X[:self.L_train]
        window_target = X[-self.L_target:]

        U_train, _, _ = self._stack_svd(window_train)
        Ur = U_train[:, :self.r]

        U_target, _, _ = self._stack_svd(window_target)
        Qm = U_target[:, :self.m]

        _, s, _ = np.linalg.svd(Ur.T.dot(Qm))

        return 1 - s[0]
