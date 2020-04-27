"""
The greedy algorithm: GreedyDUTI
Created by shouxing, 2019/6/27
"""
import numpy as np


class GreedyDUTI:
    """
    Correct the labels of the training data items using the labels of the trusted items.
    """
    def __init__(self, num_class, lam=1, max_iter=20, max_depth=20, search_grid=30, method='logistic regression'):
        """
        Init some parameter for the algorithm
        :param num_class: number of class of the data and the trusted items
        :param lam: ridge coefficient of the learner, it should be a positive real number
        :param max_iter: the maximum number of iteration
        :param max_depth: the maximum search depth.
        :param search_grid: the number of search directions (or the cluster number).
        """
        self.num_class = num_class
        self.lam = lam
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.search_grid = search_grid
        assert method == 'logistic regression' or method == 'decision tree', 'Wrong method: ' + method
        self.method = method

    def fit_transform(self, feature, label, feature_trusted, label_trusted, confidence=None):
        """
        DUTI implementation with binary RBF kernel logistic ridge regression.
        Run the duti algorithm to fit the data and the trusted item.
        :param feature: n * d ndarray, the feature vectors of the training data.
        :param label: n ndarray, the label of the training data.
        :param feature_trusted: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param label_trusted: trusted_n ndarray, the label of the trusted items.
        :param confidence: m ndarray, confidence vector of trusted items.
        :return: flag_bugs: n ndarray, boolean value, True if the label of the item is changed.
                 delta: debugging solution, n x c ndarray with value in [0,1], the sum of each row is 1.
                 ranking:bug flag ranking for prioritization, n ndarray,
                    where ranking[i] = (the iteration number when item i first have delta in [1/2, 1])
                    + (1 - that delta value)
        """
        if confidence is None:
            self.confidence = np.ones_like(label_trusted)
        else:
            self.confidence = confidence
        threshold = 0.5 # threshold for w values to treat a training point as bug
        N, D = feature.shape # the size of the training set
        trusted_N, trusted_D = feature_trusted.shape # the size of the trusted set
        assert D == trusted_D, "the dimension of the training feature should be equal to the trusted feature."

        print('training data:', N, D)
        print('num_class:', self.num_class, 'lam:', self.lam, 'max_iter:', self.max_iter,
              'max_depth:', self.max_depth, 'search_grid:', self.search_grid)

        # apapt variables
        X_train = np.column_stack((np.ones((N, 1)), feature))
        X_trust = np.column_stack((np.ones((trusted_N, 1)), feature_trusted))
        D += 1
        trusted_D += 1
        y_train = np.zeros((N, self.num_class))
        y_trust = np.zeros((trusted_N, self.num_class))
        for i in range(N):
            y_train[i][label[i]] = 1.0
        for i in range(trusted_N):
            y_trust[i][label_trusted[i]] = 1.0

        if self.method == 'logistic regression':
            self.global_theta = np.zeros((self.num_class, D))
            self.step_theta = np.zeros((self.num_class, D))
            self.global_delta = np.zeros((N, self.num_class))
            self.global_ddelta_dtheta = np.zeros((self.num_class * D, N * self.num_class))

        # find out the maximum gamma_0 value that results in a nonzero w solution,
        # i.e. \nabla_w at w = 0(i.e.the original dirty training data) and gamma = 0.
        gamma0 = 0
        delta0 = np.copy(y_train)
        if self.method == 'logistic regression':
            _, grad = self.lr_debug_object(delta0, X_train, y_train, X_trust, y_trust, gamma0)
        else:
            _, grad = self.decision_tree_debug_object(delta0, X_train, y_train, X_trust, y_trust, gamma0)
        gamma = 2 * N * max((y_train * grad).sum(axis=1))

        # Setting up parameters for fmincon
        delta = np.copy(delta0)
        ranking = np.zeros(N)
        flag_bugs = np.zeros(N) > threshold

        for iter in range(self.max_iter + 1):
            print('\nIter ----------', iter, '---------------\n')
            gamma /= 2
            if iter == self.max_iter:
                gamma = 0
            print('gamma =', gamma)
            if self.method == 'logistic regression':
                delta, step = self.gp_optimizer(lambda x: self.lr_debug_object(x, X_train, y_train, X_trust, y_trust, gamma),
                                           lambda x: self.lr_cost_object(x, X_train, y_train, X_trust, y_trust, gamma),
                                           delta, self.max_depth, self.search_grid)
            else:
                delta, step = self.gp_optimizer(
                    lambda x: self.decision_tree_debug_object(x, X_train, y_train, X_trust, y_trust, gamma),
                    lambda x: self.decision_tree_cost_object(x, X_train, y_train, X_trust, y_trust, gamma),
                    delta, self.max_depth, self.search_grid)
            violation = (y_train * (y_train - delta)).sum(axis=1)
            iter_flag_bugs = violation > 0.5
            newly_flag_bugs = iter_flag_bugs & (~ flag_bugs)
            ranking[newly_flag_bugs] = iter - violation[newly_flag_bugs]
            flag_bugs = np.copy(iter_flag_bugs)
        return flag_bugs, delta, ranking

    def lr_debug_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value and the gradient of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
                 grad: n * num_class ndarray, the gradient of all items.
        """

        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        theta0 = np.copy(self.global_theta)
        theta = self.LR_train(X_train, delta, theta0, True)
        self.step_theta = np.copy(theta)

        # get c * D implict functions G
        # compute partial derivates of Gkj w.r.t theta and delta

        dG_dtheta = np.zeros((c * D, c * D))
        dG_ddelta = np.zeros((c * D, c * N))

        P = self.lr_predict_prob(X_train, theta)

        ind_row = 0
        for k in range(c):
            # init k_col_Y
            k_col_Y = np.zeros((1, c))
            k_col_Y[0, k] = 1
            k_col_Y = np.matmul(np.ones((N, 1)), k_col_Y)

            # init k_col_P
            k_col_P = np.copy(P[:, k])

            # make common
            common_mul = 1 / N * ((k_col_Y - P).transpose() * k_col_P).transpose()

            for j in range(D):
                dgelement_dtheta = np.matmul((common_mul.transpose() * X_train[:, j]), X_train)
                dgelement_dtheta[k, j] = dgelement_dtheta[k, j] + self.lam
                dG_dtheta[ind_row, :] = dgelement_dtheta[:].transpose().reshape(c * D)

                dgelement_ddelta = np.zeros((N, c))
                dgelement_ddelta[:, k] = -1 / N * X_train[:, j]
                dG_ddelta[ind_row,:] = dgelement_ddelta[:].transpose().reshape(c * N)
                ind_row += 1

        dtheta_ddelta = - np.matmul(np.linalg.pinv(dG_dtheta), dG_ddelta)

        # gradient part trust
        P_trust = self.lr_predict_prob(X_trust, theta)
        nbla_ltrust_theta = -1 / trusted_N\
                            * np.matmul((y_trust - P_trust).transpose(),
                                        (X_trust.transpose() * self.confidence).transpose())
        dltrust_ddelta = np.matmul(dtheta_ddelta.transpose(),
                                   nbla_ltrust_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose()

        # gradient part noisy
        nabla_lnoisy_theta = -1 / N * np.matmul((delta - P).transpose(), X_train)
        dlnoisy_ddelta = np.matmul(dtheta_ddelta.transpose(), nabla_lnoisy_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose()  -1 / N * np.log(P)

        # gradient part distance
        ddist_ddelta = - gamma / N * y_train

        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) - np.mean((delta * np.log(P)).sum(axis=1)) + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1))

        grad = dltrust_ddelta + dlnoisy_ddelta + ddist_ddelta

        return cost, grad

    def lr_cost_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
        """
        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        theta0 = np.copy(self.global_theta)
        theta = self.LR_train(X_train, delta, theta0, True)

        P = self.lr_predict_prob(X_train, theta)

        # gradient part trust
        P_trust = self.lr_predict_prob(X_trust, theta)

        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) \
               - np.mean((delta * np.log(P)).sum(axis=1)) \
               + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1), dtype=np.float)

        return cost

    def LR_train(self, X, y, alpha0, preprocessed=False):
        """
        Train a LR classifier.
        :param X: n * d ndarray, the feature vectors of the training data.
        :param y: n ndarray, the current label of the training data.
        :param alpha0: num_class * d, the initial classifier.
        :param preprocessed: boolean, True if have preprocessed the input X and y.
        :return: alpha: the classifier.
        """
        N, D = X.shape
        if not preprocessed:
            X = np.column_stack((np.ones((N, 1)), X))
            y = np.zeros((N, self.num_class))
            for i in range(N):
                y[i][y[i]] = 1.0

        from scipy.optimize import minimize
        print('train a LR classifier')
        res = minimize(lambda x:self.LR_cost(x, X, y), alpha0, method='BFGS', tol=1e-6,
                       options={'maxiter': 4000, 'disp': False})
        return res['x'].reshape(self.num_class, D)

    def LR_cost(self, alpha, X, y):
        """
        Compute the value of the LR cost funciton with a given classifier
        :param alpha: num_class * d, the classifier.
        :param X: n * d ndarray, the feature vectors of the training data.
        :param y: n ndarray, the current label of the training data.
        :return: J, the value of the cost function.
        """
        N, D = X.shape
        alpha = alpha.reshape((self.num_class, D))
        h = np.exp(np.matmul(X, alpha.transpose()))
        sum_h = h.sum(axis=1)
        h = (h.transpose() / sum_h).transpose()
        J_err = -np.mean((y * np.log(h)).sum(axis=1))
        J_reg = self.lam / 2 * (alpha * alpha).sum()
        J = J_err + J_reg
        return J

    def lr_predict_prob(self, X, theta):
        """
        Predict the label distribution of input items.
        :param X: n * d ndarray, the feature vectors of the input items.
        :param theta: num_class * d, the parameter of the classifier.
        :return: h: n * num_class ndarray, the label distribution of input items.
        """
        h = np.exp(np.matmul(X, theta.transpose()))
        sum_h = h.sum(axis=1)
        h = (h.transpose() / sum_h).transpose()
        return h

    def decision_tree_train(self, X, y):
        from sklearn import tree
        index = np.argmax(y, axis=1)
        N, D = y.shape
        y = np.zeros_like(y, dtype=np.int).tolist()
        for i in range(N):
            y[i][index[i]] = 1
        y = np.array(y, dtype=np.int)
        clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, min_samples_split=5)
        print(clf)
        clf.fit(X, y)
        return clf

    def decision_tree_predict_prob(self, X, clf):
        return clf.predict(X)

    def decision_tree_debug_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value and the gradient of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
                 grad: n * num_class ndarray, the gradient of all items.
        """

        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        clf = self.decision_tree_train(X_train, delta)

        # get c * D implict functions G
        # compute partial derivates of Gkj w.r.t theta and delta

        dG_dtheta = np.zeros((c * D, c * D))
        dG_ddelta = np.zeros((c * D, c * N))

        P = self.decision_tree_predict_prob(X_train, clf)
        if self.method == 'decision tree':
            P = P * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps

        ind_row = 0
        for k in range(c):
            # init k_col_Y
            k_col_Y = np.zeros((1, c))
            k_col_Y[0, k] = 1
            k_col_Y = np.matmul(np.ones((N, 1)), k_col_Y)

            # init k_col_P
            k_col_P = np.copy(P[:, k])

            # make common
            common_mul = 1 / N * ((k_col_Y - P).transpose() * k_col_P).transpose()

            for j in range(D):
                dgelement_dtheta = np.matmul((common_mul.transpose() * X_train[:, j]), X_train)
                dgelement_dtheta[k, j] = dgelement_dtheta[k, j] + self.lam
                dG_dtheta[ind_row, :] = dgelement_dtheta[:].transpose().reshape(c * D)

                dgelement_ddelta = np.zeros((N, c))
                dgelement_ddelta[:, k] = -1 / N * X_train[:, j]
                dG_ddelta[ind_row,:] = dgelement_ddelta[:].transpose().reshape(c * N)
                ind_row += 1

        dtheta_ddelta = - np.matmul(np.linalg.pinv(dG_dtheta), dG_ddelta)

        # gradient part trust
        P_trust = self.decision_tree_predict_prob(X_trust, clf)
        if self.method == 'decision tree':
            P_trust = P_trust * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
        nbla_ltrust_theta = -1 / trusted_N\
                            * np.matmul((y_trust - P_trust).transpose(),
                                        (X_trust.transpose() * self.confidence).transpose())
        dltrust_ddelta = np.matmul(dtheta_ddelta.transpose(),
                                   nbla_ltrust_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose()

        # gradient part noisy
        nabla_lnoisy_theta = -1 / N * np.matmul((delta - P).transpose(), X_train)


        dlnoisy_ddelta = np.matmul(dtheta_ddelta.transpose(),
                                   nabla_lnoisy_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose() \
                         - 1 / N * np.log(P)

        # gradient part distance
        ddist_ddelta = - gamma / N * y_train

        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) - np.mean((delta * np.log(P)).sum(axis=1)) + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1))

        grad = dltrust_ddelta + dlnoisy_ddelta + ddist_ddelta

        return cost, grad

    def decision_tree_cost_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
        """
        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        clf = self.decision_tree_train(X_train, delta)

        P = self.decision_tree_predict_prob(X_train, clf)

        # gradient part trust
        P_trust = self.decision_tree_predict_prob(X_trust, clf)

        if self.method == 'decision tree':
            P = P * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
            P_trust = P_trust * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) \
               - np.mean((delta * np.log(P)).sum(axis=1)) \
               + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1), dtype=np.float)

        return cost

    def decision_tree_train(self, X, y):
        from sklearn import tree
        index = np.argmax(y, axis=1)
        N, D = y.shape
        y = np.zeros_like(y, dtype=np.int).tolist()
        for i in range(N):
            y[i][index[i]] = 1
        y = np.array(y, dtype=np.int)
        clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, min_samples_split=5)
        print(clf)
        clf.fit(X, y)
        return clf

    def decision_tree_predict_prob(self, X, clf):
        return clf.predict(X)

    def decision_tree_debug_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value and the gradient of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
                 grad: n * num_class ndarray, the gradient of all items.
        """

        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        clf = self.decision_tree_train(X_train, delta)

        # get c * D implict functions G
        # compute partial derivates of Gkj w.r.t theta and delta

        dG_dtheta = np.zeros((c * D, c * D))
        dG_ddelta = np.zeros((c * D, c * N))

        P = self.decision_tree_predict_prob(X_train, clf)
        if self.method == 'decision tree':
            P = P * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps

        ind_row = 0
        for k in range(c):
            # init k_col_Y
            k_col_Y = np.zeros((1, c))
            k_col_Y[0, k] = 1
            k_col_Y = np.matmul(np.ones((N, 1)), k_col_Y)

            # init k_col_P
            k_col_P = np.copy(P[:, k])

            # make common
            common_mul = 1 / N * ((k_col_Y - P).transpose() * k_col_P).transpose()

            for j in range(D):
                dgelement_dtheta = np.matmul((common_mul.transpose() * X_train[:, j]), X_train)
                dgelement_dtheta[k, j] = dgelement_dtheta[k, j] + self.lam
                dG_dtheta[ind_row, :] = dgelement_dtheta[:].transpose().reshape(c * D)

                dgelement_ddelta = np.zeros((N, c))
                dgelement_ddelta[:, k] = -1 / N * X_train[:, j]
                dG_ddelta[ind_row,:] = dgelement_ddelta[:].transpose().reshape(c * N)
                ind_row += 1

        dtheta_ddelta = - np.matmul(np.linalg.pinv(dG_dtheta), dG_ddelta)

        # gradient part trust
        P_trust = self.decision_tree_predict_prob(X_trust, clf)
        if self.method == 'decision tree':
            P_trust = P_trust * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
        nbla_ltrust_theta = -1 / trusted_N\
                            * np.matmul((y_trust - P_trust).transpose(),
                                        (X_trust.transpose() * self.confidence).transpose())
        dltrust_ddelta = np.matmul(dtheta_ddelta.transpose(),
                                   nbla_ltrust_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose()

        # gradient part noisy
        nabla_lnoisy_theta = -1 / N * np.matmul((delta - P).transpose(), X_train)


        dlnoisy_ddelta = np.matmul(dtheta_ddelta.transpose(),
                                   nabla_lnoisy_theta[:].transpose().reshape(c * D)).reshape(c, N).transpose() \
                         - 1 / N * np.log(P)

        # gradient part distance
        ddist_ddelta = - gamma / N * y_train

        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) - np.mean((delta * np.log(P)).sum(axis=1)) + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1))

        grad = dltrust_ddelta + dlnoisy_ddelta + ddist_ddelta

        return cost, grad

    def decision_tree_cost_object(self, delta, X_train, y_train, X_trust, y_trust, gamma):
        """
        Train a classifier with the current label of the training items,
        and then compute the value of the cost function according to
        the label of the training items and the trusted items.
        :param delta: n * num_class ndarray, the current label distribution of all items.
        :param X_train: n * d ndarray, the feature vectors of the training data.
        :param y_train: n ndarray, the label of the training data.
        :param X_trust: trusted_n * d ndarray, the feature vectors of the trusted items.
        :param y_trust: trusted_n ndarray, the label of the trusted items.
        :param gamma: the weight of the normalization part in the minimization problem.
        :return: cost: the value of the cost function.
        """
        N, D = X_train.shape
        trusted_N, trusted_D = X_trust.shape
        _, c = delta.shape
        # find the point for a given delta (delta, theta)
        # theta0 = zeros(c, d);
        clf = self.decision_tree_train(X_train, delta)

        P = self.decision_tree_predict_prob(X_train, clf)

        # gradient part trust
        P_trust = self.decision_tree_predict_prob(X_trust, clf)

        if self.method == 'decision tree':
            P = P * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
            P_trust = P_trust * (1 - np.finfo(float).eps * 2) + np.finfo(float).eps
        cost = - np.mean((y_trust * np.log(P_trust)).sum(axis=1) * self.confidence) \
               - np.mean((delta * np.log(P)).sum(axis=1)) \
               + gamma * np.mean((y_train - (y_train * delta)).sum(axis=1), dtype=np.float)

        return cost

    def gp_optimizer(self, func, cost_func, delta0, max_depth, search_grid):
        """
        Search for a best solution for the problem.
        :param func: a classification model, return the value of the cost function and the gradient.
        :param cost_func: a classification model, return the value of the cost function.
        :param delta0: n * num_class ndarray, the label distribution of all items.
        :param max_depth: the maximum search depth.
        :param search_grid: the number of search directions (or the cluster number).
        :return: delta: n * num_class ndarray, the label distribution of all items be found.
                 step: n * num_class ndarray, the delta of the label distribution for all items.
        """
        for i in range(max_depth):
            flag_continue, next_delta, cost = self.line_search(func, cost_func, delta0, search_grid, i)
            if flag_continue:
                delta0 = np.copy(next_delta)
            else:
                break

        delta = np.copy(delta0)
        step = delta - delta0
        return delta, step

    def row_direction_search(self, grad, delta, D):
        """
        For each item, it would do the maximum change in the direction of the gradient under the constraints,
        or just keep the distribution stable.
        :param grad: num_class ndarray, the gradient of the item.
        :param delta: num_class ndarray, the label distribution of the item.
        :param D: number of class
        :return: flag_feasible: boolean, True if the item would change.
                 step: num_class ndarray, the delta of the label distribution.
                 grad_score: float, the gain by the change of item.
        """
        step = -grad

        # project step
        dim = D
        non_decreasabe_entry = (delta <= np.finfo(float).eps)
        fixed_all = np.zeros(D, dtype=np.bool)
        while True:
            fixed_entry = (~fixed_all) & non_decreasabe_entry & (step < sum(step) / dim)
            step[fixed_entry] = 0

            fixed_all = fixed_all | fixed_entry
            n_dim = D - sum(fixed_all)
            if (n_dim == dim) | (n_dim == 1):
                dim = n_dim
                break
            dim = n_dim

        if dim == 1:
            return False, np.zeros((1, D)), 0

        flag_feasible = True
        # dim
        mutable = ~fixed_all
        # mutable
        step[mutable] = step[mutable] - step.sum() / dim
        grad_score = -1 * (step * grad).sum()
        # step
        out_bounded = step < 0
        scale = min(-delta[out_bounded] / step[out_bounded])
        step = step * scale
        return flag_feasible, step, grad_score

    def gp_step(self, grad0, delta0, N, D):
        """
        Choose some items to do the maximum change in the direction of the gradient under the constraints,
        and the others just keep the distribution stable.
        :param grad0: n * num_class ndarray, the gradients of all items.
        :param delta0: n * num_class ndarray, the label distribution of all items.
        :param N: number of items.
        :param D: number of class.
        :return: flag_feasible: n ndarray, boolean, True if the item would change.
                 step: n * num_class ndarray, the delta of the label distribution for all items.
                 grad_score: n ndarray, float, the gain by the change of all items.
        """
        step = np.zeros((N, D))
        flag_feasible = np.zeros(N, dtype=np.bool)
        grad_score = np.zeros(N)
        for i in range(N):
            i_flag_feasible, i_maximum_step, i_grad_score = self.row_direction_search(grad0[i, :], delta0[i, :], D)
            flag_feasible[i] = i_flag_feasible
            step[i, :] = np.copy(i_maximum_step)
            grad_score[i] = i_grad_score
        return flag_feasible, step, grad_score

    def line_search(self, func, cost_func, delta0, search_grid, depth):
        """
        Find a better label distribution according to the gradient.
        :param func: a classification model, return the value of the cost function and the gradient.
        :param cost_func: a classification model, return the value of the cost function.
        :param delta0: n * num_class ndarray, the label distribution of all items.
        :param search_grid: the number of search directions (or the cluster number).
        :param depth: the search depth.
        :return: flag_continue: boolean, True if found a better label distribution.
                 delta: n * num_class ndarray, the better label distribution of all items be found.
                 cost: the value of the cost function under the new label distribution.
        """
        flag_continue = False

        N, D = delta0.shape

        cost0, grad0 = func(delta0)
        if self.method == 'logistic regression':
            self.global_theta = np.copy(self.step_theta)
        min_group = 10

        # line search the best based on current
        # deal with some special cases
        flag_feasible, maximum_step, grad_score = self.gp_step(grad0, delta0, N, D)
        num_feasible = flag_feasible.sum()

        # if num_feasible <= min_group
        if num_feasible <= min_group:
            delta_to_test = np.copy(delta0)
            delta_to_test[flag_feasible, :] += maximum_step[flag_feasible, :]
            cost = cost_func(delta_to_test)
            if cost < cost0:
                print('Searching depth', depth, 'Cost', cost0, '-->', cost)
                return True, delta_to_test, cost
            else:
                print('Searching depth', depth, 'end.')
                return False, delta0, cost0

        thresholds = self.calc_search_grid_thresholds(grad_score[flag_feasible])

        last_num_changed = -1
        best_cost = cost0
        delta = np.copy(delta0)

        for i in range(len(thresholds)):
            grad_threshold = thresholds[i]
            flag_to_test = grad_score >= grad_threshold
            num_to_change = flag_to_test.sum()
            if num_to_change == last_num_changed:
                continue
            last_num_changed = num_to_change
            delta_to_test = np.copy(delta0)
            delta_to_test[flag_to_test, :] += maximum_step[flag_to_test, :]
            cost = cost_func(delta_to_test)
            print('Searching depth', depth, 'grid:', i, '/', search_grid, ', cost:', cost, ', num changed:', num_to_change, '/', num_feasible)
            if cost <= best_cost:
                best_cost = cost
                delta = np.copy(delta_to_test)
        flag_continue = best_cost < cost0
        cost = best_cost
        if flag_continue:
            print('Searching depth', depth, 'Cost', cost0, '-->', cost)
        else:
            print('Searching depth', depth, 'end')
        return flag_continue, delta, cost

    def calc_search_grid_thresholds(self, grad_scores):
        """
        Group items according to its gain.
        :param grad_scores: n ndarray, the gain of items.
        :return: group_number ndarray, thresholds to group items
        """
        policy = 'min_ranges'
        if policy == 'avg_scores':
            min_score = grad_scores.min()
            max_score = grad_scores.max()
            thresholds = min_score + (max_score - min_score) * np.linspace(0, 1, self.search_grid)
            return thresholds

        if policy == 'min_ranges':
            l = len(grad_scores)
            if l <= self.search_grid:
                thresholds = np.sort(grad_scores)
                return thresholds
            asending_scores_lower = np.sort(grad_scores)
            asending_scores_higher = np.zeros(l)
            asending_scores_higher[: l - 1] = asending_scores_lower[1: l]
            gaps = asending_scores_higher - asending_scores_lower
            max_gap_idx = np.argsort(-gaps)
            max_gap_idx = max_gap_idx[0: self.search_grid - 1]
            thresholds = np.zeros(self.search_grid)
            thresholds[0] = grad_scores.min()
            thresholds[1:] = (asending_scores_higher[max_gap_idx] + asending_scores_lower[max_gap_idx]) / 2
            thresholds[1:] = np.sort(thresholds[1:])
            return thresholds


if __name__ == '__main__':
    from src import DataLoader, ResultSaver
    from scipy.io import loadmat

    data = loadmat('../DUTI_code/dataset.mat')
    trust_item = loadmat('../DUTI_code/trust_item.mat')

    trust_item_index = trust_item['indexs'].reshape(-1)[:10]
    trust_item_label = trust_item['label'].reshape(-1)[:10]
    print(np.max(trust_item_index))
    feature = data['feature'][:650]
    label = data['label'].reshape(-1)[:650]
    trusted_feature = feature[trust_item_index]

    duti = GreedyDUTI(num_class=8, search_grid=10)
    bugs, delta, rankings = duti.fit_transform(feature, label, trusted_feature, trust_item_label)
    y_debug = np.copy(label)
    clean_bug_y = np.argmax(delta[bugs, :], axis=1)
    y_debug[bugs] = clean_bug_y
    y_debug[trust_item_index] = trust_item_label
    result = []
    for i, y in enumerate(y_debug):
        result.append({
            'id': i,
            'label_name': y,
            'flag': 0
        })
    ResultSaver.save_as_txt('../result/dataset.txt', result)