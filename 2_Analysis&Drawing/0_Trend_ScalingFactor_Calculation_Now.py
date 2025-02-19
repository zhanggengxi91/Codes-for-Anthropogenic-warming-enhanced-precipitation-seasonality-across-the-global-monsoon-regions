# -- coding:utf-8 --
import pandas as pd
import numpy as np
import os
from glob import glob
import scipy.stats as stats
from scipy.stats import linregress

dir = 'G:/1_OrigionalData/5_Attribution/2_csv_1985-2014/'

#######                                  Module 1 utils                                             ################
def speco(C):
    """
        This function computes eigenvalues and eigenvectors, in descending order
        :param C: numpy.ndarray
            A p x p symetric real matrix
        :return:
        P: numpy.ndarray
            The eigenvectors (P[:, i] is the ist eigenvector)
        D: numpy.ndarray
            The eigenvalues as a diagonal matrix
        """
    # Compute eigenvalues and eigenvectors (the eigenvectors are non unique so the values may change from one software
    # to another e.g. python, matlab, scilab)
    D0, P0 = np.linalg.eig(C)
    # Take real part (to avoid numeric noise, eg small complex numbers)
    if np.max(np.imag(D0)) / np.max(np.real(D0)) > 1e-12:
        raise ValueError("Matrix is not symmetric")

    # Check that C is symetric (<=> real eigen-values/-vectors)
    P1 = np.real(P0)
    D1 = np.real(D0)

    # sort eigenvalues in descending order and
    # get their indices to order the eigenvector
    Do = np.sort(D1)[::-1]
    o = np.argsort(D1)[::-1]
    P = P1[:, o]
    D = np.diag(Do)
    return P, D


def chi2_test(d_cons, df):
    """
       Check whether it is from a chi-squared distribution or not
       :param d_cons: float
           -2 log-likelihood
       :param df: int
           Degrees of freedom
       :return:
       pv_cons: float
           p-value for the test
       """
    rien = stats.chi2.cdf(d_cons, df=df)
    pv_cons = 1. - rien
    return pv_cons


def project_vectors(nt, X):
    """
        This function provides a projection matrix U that can be applied to X to ensure its covariance matrix to be
        full-ranked. Projects to a nt-1 subspace (ref: Ribes et al., 2013).
        :param nt: int
            number of time steps
        :param X: numpy.ndarray
            nt x nf array to be projected
        :return:
        np.dot(U, X): numpy.ndarray
            nt - 1 x nf array of projected timeseries
        """
    M = np.eye(nt, nt) - np.ones((nt, nt)) / nt
    # Eigen-vectors/-values of M; note that rk(M)=nt-1, so M has one eigenvalue equal to 0.
    u, d = speco(M)
    # (nt-1) first eigenvectors (ie the ones corresponding to non-zero eigenvalues)
    U = u[:, :nt - 1].T
    return np.dot(U, X)


def unproject_vectors(nt, Xc):
    """
       This function provides unprojects a matrix nt subspace to we can compute the trends
       :param nt: int
           number of time steps
       :param Xc: numpy.ndarray
           nt x nf array to be unprojected
       :return:
       np.dot(U, X): numpy.ndarray
           nt - 1 x nf array of projected timeseries
       """
    M = np.eye(nt, nt) - np.ones((nt, nt)) / nt
    # Eigen-vectors/-values of M; note that rk(M)=nt-1, so M has one eigenvalue equal to 0.
    u, d = speco(M)
    # inverse of the projection matrix
    Ui = np.linalg.inv(u.T)[:, :nt - 1]
    return np.dot(Ui, Xc)


def SSM(exp, X_mm, domain, jijie="ANN", zhishu="rx1day"):
    """
        Calculates the squared difference between each models ensemble mean and the multi-model mean. Based on
        (Ribes et al., 2017)
        :param exp: str
            Experiment to calculate the difference (e.g., 'historical', 'historicalNat')
        :param X_mm: numpy.ndarray
            Array with multi-model ensemble mean
        :param init: int
            Correspondent year to start the analysis
        :param end: int
            Correspondent year to finish the analysis
        :return:
        np.diag(((Xc - Xc_mm) ** 2.).sum(axis=1)): numpy.ndarray
            nt -1 x nt - 1 array of the difference between each model ensemble mean the multi-model mean
        """
    # reads ensemble mean for each model
    ifiles = glob(dir + jijie + '/model/%s/ensmean/%s@model@%s@ensmean@%s@%s@*.csv' % (exp, jijie, exp, domain, zhishu))
    df = pd.DataFrame()
    for ifile in ifiles:
        df_temp = pd.read_csv(ifile, index_col=0, parse_dates=True)
        df = pd.concat([df, df_temp['0'].to_frame(os.path.basename(ifile)[:-4])], axis=1)
    # remove columns (ensemble members with nan)
    df.dropna(inplace=True, axis=1)
    # gets ensemble values and multi model (mm) ensemble
    X = df.values
    # project the data
    Xc = project_vectors(X.shape[0], X)
    Xc_mm = project_vectors(X.shape[0], X_mm.reshape((X.shape[0], 1)))
    return np.diag(((Xc - Xc_mm) ** 2.).sum(axis=1))


def get_nruns(exp, domain, how='pandas', jijie="ANN", zhishu="rx1day"):
    """
        Reads the number of runs for each model
        :param exp: str
            Experiment to calculate the difference (e.g., 'historical', 'historicalNat')
        :param how: str
            Used to see if the number of runs is calculated using the pandas dataframes or text file ('historicalOA' for
            example)
        :param init: int
            Correspondent year to start the analysis
        :param end: int
            Correspondent year to finish the analysis
        :return:
        nruns: numpy.ndarray
           Array with the number of runs for each model
        """
    global nruns
    if how == 'pandas':
        ifiles = glob(dir + jijie + '/model/%s/ensmean/%s@model@%s@ensmean@%s@%s@*.csv' % (exp, jijie, exp, domain, zhishu))
        nruns = []

        for ifile in sorted(ifiles):
            df_temp = pd.read_csv(ifile, index_col=0, parse_dates=True)
            nruns.append(len(df_temp.columns))

        nruns = np.array(nruns)
    return nruns


def Cm_estimate(exp, Cv, X_mm, domain, how_nr='pandas', jijie="ANN", zhishu="rx1day"):
    """
        Estimated covariance matrix for model error (Ribes et al., 2017)
        :param exp: str
            Experiment to calculate the difference (e.g., 'historical', 'historicalNat')
        :param Cv: numpy.ndarray
            Array with internal variability covariance matrix
        :param X_mm: numpy.ndarray
            Array with multi-model ensemble mean
        :param how_nr:
            Used to see if the number of runs is calculated using the pandas dataframes or text file ('historicalOA' for
            example)
        :param init: int
            Correspondent year to start the analysis
        :param end: int
            Correspondent year to finish the analysis
        :return:
        Cm_pos_hat: numpy.ndarray
            Estimated covariance matrix for model error
        """

    # model difference
    _SSM = SSM(exp, X_mm, domain=domain, jijie=jijie, zhishu=zhishu)
    # nruns - number of runs / nm - number of models
    nruns = get_nruns(exp, domain=domain, how=how_nr, jijie=jijie, zhishu=zhishu)
    nm = len(nruns)
    Cv_all = np.zeros(Cv.shape)
    for nr in nruns:
        Cv_all += Cv / nr
    # first estimation of Cm
    Cm_hat = (1. / (nm - 1.)) * (_SSM - ((nm - 1.) / nm) * Cv_all)
    # set negative eigenvalues to zero and recompose the signal
    S, X = np.linalg.eig(Cm_hat)
    S[S < 0] = 0
    Cm_pos_hat = np.linalg.multi_dot([X, np.diag(S), np.linalg.inv(X)])  # spectral decomposition
    Cm_pos_hat = (1. + (1. / nm)) * Cm_pos_hat
    return Cm_pos_hat


def Cv_estimate(exp, Cv, domain, how_nr='pandas', jijie="ANN", zhishu="rx1day"):
    """
       Estimated covariance matrix for internal variability considering multiple models (Ribes et al., 2017)
       :param exp: str
           Experiment to calculate the difference (e.g., 'historical', 'historicalNat')
       :param Cv: numpy.ndarray
           Array with internal variability covariance matrix
       :param how_nr:
           Used to see if the number of runs is calculated using the pandas dataframes or text file ('historicalOA' for
           example)
       :param init: int
           Correspondent year to start the analysis
       :param end: int
           Correspondent year to finish the analysis
       :return:
       Cv_estimate: numpy.ndarray
           Estimated covariance matrix for internal variability considering multiple models
       """
    # nruns - number of runs / nm - number of models
    nruns = get_nruns(exp, domain=domain, how=how_nr, jijie=jijie, zhishu=zhishu)
    nm = len(nruns)
    Cv_all = np.zeros(Cv.shape)
    for nr in nruns:
        Cv_all += Cv / nr
    Cv_estimate = (1. / (nm ** 2.) * Cv_all)
    return Cv_estimate

########                              Module 2 trend calculate                         ######################
def calculate_trend(y):
    """
       Calculate the trend by unprojecting a vector to the nt subspace and the using OLS estimation
       :param y: numpy.ndarray
           nt -1 array used to calculate the trend
       :return:
       beta_hat: numpy.ndarray
           value of the scaling factor of the OLS adjustment
       """
    nt = len(y) + 1
    y_un = unproject_vectors(nt, y)  # unproject the data
    X = np.vstack([np.ones(nt), np.arange(nt)]).T
    # use OLS to calculate the trend
    beta_hat = np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([X.T, X])), X.T, y_un])
    return beta_hat[1]  # only return the trend


def calculate_uncertainty(y, Cy, alpha=0.1, nsamples=4000):
    """
       Calculate trend uncertainty by generating multiple series and the calculating the confidence interval
       :param y: numpy.ndarray
           nt -1 array used to calculate the trend
       :param Cy: numpy.ndarray
           nt -1 x nt -1 covariance matrix from the y vector
       :param alpha: float
           significance level
       :param nsamples: int
           number of repetitions
       :return:
       np.array([trend_min, trend_max]): np.ndarray
           array with the minimum and maximum values from the confidence interval
       """
    trends = np.zeros(nsamples)
    for i in range(nsamples):
        y_random = np.random.multivariate_normal(y, Cy)  # generate random vector based on the mean and cov matrix
        # calculate the trend
        trends[i] = calculate_trend(y_random)
    trend_min = np.percentile(trends, (alpha * 100) / 2.)
    trend_max = np.percentile(trends, 100 - (alpha * 100) / 2.)
    return np.array([trend_min, trend_max])


def all_trends(y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat):
    """
       Calculate all trends (observations and each individual forcing) and save it to a csv file
       :param y_star_hat: np.ndarray
           vector of observations
       :param Xi_star_hat: np.ndarray
           nt -1 x nf matrix of forcings where nt is the number of time steps and nf is the number of forcings
       :param Cy_star_hat: np.ndarray
           nt -1 x nt -1 covariance matrix for the observations
       :param Cxi_star_hat: np.ndarray
           nf x nt -1 x nt -1 covariance matrix for each individual forcing
       :return:
       df: pandas.DataFrame
           dataframe with the trend for the observations and each of the forcings
       """
    trends_list = []
    trend = calculate_trend(y_star_hat)
    confidence_interval = calculate_uncertainty(y_star_hat, Cy_star_hat, alpha=0.1)
    trends_list.append(['Observation', trend, confidence_interval[0], confidence_interval[1]])
    # print('-' * 60)
    # print('Trends from the analysis ...')
    # print('%30s: %.3f (%.3f, %.3f)' % ('Observation', trend, confidence_interval[0], confidence_interval[1]))
    nf = Xi_star_hat.shape[1]
    for i in range(nf):
        trend = calculate_trend(Xi_star_hat[:, i])
        confidence_interval = calculate_uncertainty(Xi_star_hat[:, i], Cxi_star_hat[i], alpha=0.1)
        # print('%30s: %.3f (%.3f, %.3f)' % ('Forcing no %d only' % (i + 1), trend, confidence_interval[0],
        #                                    confidence_interval[1]))
        trends_list.append(['Forcing no %d only' % (i + 1), trend, confidence_interval[0], confidence_interval[1]])
    # save data as csv
    df = pd.DataFrame(trends_list, columns=['forcing', 'trend', 'trend_min', 'trend_max'])
    # df.to_csv('trends.csv', index=False)
    return df

#####                              Module 3 preprocess                             ###############
class PreProcess:
    """
        A class to preprocess model data to be used in attribution methods like OLS, TLS and others. The script is
        heavily based on Aurelien Ribes (CNRM-GAME) scilab code, so for more information the user should consult the
        following reference:
            Ribes A., S. Planton, L. Terray (2012) Regularised optimal fingerprint for attribution.
            Part I: method, properties and idealised analysis. Climate Dynamics.
        :attribute y: numpy.ndarray
            Array of size nt with observations as a timeseries
        :attribute X: numpy.ndarray
            Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
        :attribute Z: numpy.ndarray
            Array of ensembles with internal variability used to compute covariance matrix
        :attribute nt: int
            Number of timesteps for the timeseries
        :method extract_Z2(self, method='regular', frac=0.5):
            Split big sample Z in Z1 and Z2
        :method proj_fullrank(self, Z1, Z2):
            Provides a projection matrix to ensure its covariance matrix to be full-ranked
        :method creg(self, X, method='ledoit', alpha1=1e-10, alpha2=1):
            Computes regularized covariance matrix
    """

    def __init__(self, y, X, Z):
        """
               :param y: numpy.ndarray
                   Array of size nt with observations as a timeseries
               :param X: numpy.ndarray
                   Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
               :param Z: numpy.ndarray
                   Array of ensembles with internal variability used to compute covariance matrix
        """
        self.y = y
        self.X = X
        self.Z = Z
        self.nt = y.shape[0]

    def extract_Z2(self, method='regular', frac=0.5):
        """
               This function is used to split a big sample Z (dimension: nz x p, containing nz iid realisation of a random
               vector of size p) into two samples Z1 and Z2 (respectively of dimension nz1 x p and nz2 x p, with
               nz = nz1 + nz2). Further explanations in Ribes et al. (2012).
               :param method: str
                   type of sampling used, for now may be only 'regular'
               :param frac: float
                   fraction of realizations to put in Z2, the remaining is used in Z1
               :return:
               Z1: numpy.ndarray
                   Array of size (nz1 x p)
               Z2: numpy.ndarray
                   Array of size (nz2 x p)
        """
        nz = self.Z.shape[1]
        ind_z2 = np.zeros(nz)
        if method == 'regular':
            ix = np.arange(1 / frac - 1, nz, 1 / frac).astype(int)  # -1 is because python starts index at 0
            ind_z2[ix] = 1
            Z2 = self.Z[:, ind_z2 == 1]
            Z1 = self.Z[:, ind_z2 == 0]
        else:
            raise NotImplementedError('Method not implemented yet')

        return Z1, Z2

    def proj_fullrank(self, Z1, Z2):
        """
                This function provides a projection matrix U that can be applied to y, X, Z1 and Z2 to ensure its covariance
                matrix to be full-ranked. Uses variables defined in 'self', Z1 and Z2 computed in 'extract_Z2' method.
                :param Z1: numpy.ndarray
                    Array of size (nz1 x p) of control simulation
                :param Z2: numpy.ndarray
                    Array of size (nz1 x p) of control simulation
                :return:
                yc: numpy.ndarray
                    y projected in U
                Xc: numpy.ndarray
                    X projected in U
                Z1: numpy.ndarray
                    Z1 projected in U
                Z2c: numpy.ndarray
                    Z2 projected in U
                """
        # M: the matrix corresponding to the temporal centering
        M = np.eye(self.nt, self.nt) - np.ones((self.nt, self.nt)) / self.nt
        # Eigen-vectors/-values of M; note that rk(M)=nt-1, so M has one eigenvalue equal to 0.
        u, d = speco(M)
        # (nt-1) first eigenvectors (ie the ones corresponding to non-zero eigenvalues)
        U = u[:, :self.nt - 1].T
        # Project all input data
        yc = np.dot(U, self.y)
        Xc = np.dot(U, self.X)
        Z1c = np.dot(U, Z1)
        Z2c = np.dot(U, Z2)
        return yc, Xc, Z1c, Z2c

    def creg(self, X, method='ledoit', alpha1=1e-10, alpha2=1):
        """
               This function compute the regularised covariance matrix estimate following the equation
               'Cr = alpha1 * Ip + alpha2 * CE' where alpha1 and alpha2 are parameters Ip is the p x p identity matrix and CE
               is the sample covariance matrix
               :param X: numpy.ndarray
                   A n x p sample, meaning n iid realization of a random vector of size p.
               :param method: str
                   method to compute the regularized covariance matrix
                   - 'ledoit' uses Ledoit and Wolf (2003) estimate (default)
                   - 'specified' uses specified values of alpha1 and alpha2
               :param alpha1: float
                   Specified value for alpha1 (not used if method different than 'specified')
               :param alpha2: float
                   Specified value for alpha1 (not used if method different than 'specified')
               :return:
               Cr: numpy.ndarray
                   Regularized covariance matrix
               """
        n, p = X.shape
        CE = np.dot(X.T, X) / n  # sample covariance
        Ip = np.eye(p, p)
        # method for the regularised covariance matrix estimate as introduced by Ledoit & Wolf (2004) more specifically
        # on pages 379-380
        if method == 'ledoit':
            m = np.trace(np.dot(CE, Ip)) / p  # first estimate in L&W
            XP = CE - m * Ip
            d2 = np.trace(np.dot(XP, XP.T)) / p  # second estimate in L&W
            bt = np.zeros(n)
            for i in range(n):
                Xi = X[i, :].reshape((1, p))
                Mi = np.dot(Xi.T, Xi)
                bt[i] = np.trace(np.dot((Mi - CE), (Mi - CE).T)) / p
            bb2 = (1. / n ** 2.) * bt.sum()
            b2 = min(bb2, d2)  # third estimate in L&W
            a2 = d2 - b2  # fourth estimate in L&W
            alpha1 = (b2 * m / d2)
            alpha2 = (a2 / d2)
        elif method != 'specified':
            raise NotImplementedError('Method not implemented yet')
        Cr = alpha1 * Ip + alpha2 * CE
        return Cr


#####################               Module 4 Attribution models                          ###############################
class AttributionModel:
    """
       A class for attribution models. The OLS implementation is heavily based on Aurelien Ribes (CNRM-GAME) scilab code
       (see more in 'preprocess.py'). Also, Aurelien Ribes model proposed in 2017 is implemented following the reference:
           Ribes, Aurelien, et al. (2017) A new statistical approach to climate change detection and attribution.
           Climate Dynamics.
       :attribute X: numpy.ndarray
           Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
       :attribute y: numpy.ndarray
           Array of size nt with observations as a timeseries
       :method ols(self, Cf, Proj, Z2, cons_test='AT99'):
           Ordinary Least Square (OLS) estimation of beta from the linear model y = beta * X + epsilon
       """

    def __init__(self, X, y):
        """
                :param X: numpy.ndarray
                    Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
                :param y: numpy.ndarray
                    Array of size nt with observations as a timeseries
                """
        self.y = y
        self.X = X
        self.nt = y.shape[0]
        self.nr = self.nt - 1  # 1 stands for the number of spatial patterns (dealing only with timeseries)
        self.I = X.shape[1]

    def ols(self, Cf, Proj, Z2, cons_test='AT99'):
        """
               Ordinary Least Square (OLS) estimation of beta from the linear model y = beta * X + epsilon as discussed in the
               following reference:
                   Allen, Myles R., and Simon FB Tett (1999) Checking for model consistency in optimal fingerprinting.
                   Climate Dynamics.
               :param Cf: numpy.ndarray
                   Covariance matrix. Be sure that Cf is invertible to use this model (look at PreProcess class)
               :param Proj: numpy.ndarray
                   Array of zeros and ones, indicating which forcings in each simulation
               :param Z2: numpy.ndarray
                   Array of size (nz1 x p) of control simulation used to compute consistency test
               :param cons_test: str
                   Which consistency test to be used
                   - 'AT99' the formula provided by Allen & Tett (1999) (default)
               :return:
               Beta_hat: dict
                   Dictionary with estimation of beta_hat and the upper and lower confidence intervals
               """
        # computes the covariance inverse
        Cf1 = np.linalg.inv(Cf)
        _Ft = np.linalg.multi_dot([self.X.T, Cf1, self.X])
        _Ft1 = np.linalg.inv(_Ft)
        Ft = np.linalg.multi_dot([_Ft1, self.X.T, Cf1]).T
        _y = self.y.reshape((self.nt, 1))
        beta_hat = np.linalg.multi_dot([_y.T, Ft, Proj.T])
        # 1-D confidence interval
        nz2 = Z2.shape[1]
        Z2t = Z2.T
        Var_valid = np.dot(Z2t.T, Z2t) / nz2
        Var_beta_hat = np.linalg.multi_dot([Proj, Ft.T, Var_valid, Ft, Proj.T])
        beta_hat_inf = beta_hat - 2. * stats.t.cdf(0.90, df=nz2) * np.sqrt(np.diag(Var_beta_hat).T)
        beta_hat_sup = beta_hat + 2. * stats.t.cdf(0.90, df=nz2) * np.sqrt(np.diag(Var_beta_hat).T)
        # consistency check
        epsilon = _y - np.linalg.multi_dot([self.X, np.linalg.inv(Proj), beta_hat.T])
        if cons_test == 'AT99':  # formula provided by Allen & Tett (1999)
            d_cons = np.linalg.multi_dot([epsilon.T, np.linalg.pinv(Var_valid), epsilon]) / (self.nr - self.I)
            rien = stats.f.cdf(d_cons, dfn=self.nr - self.I, dfd=nz2)
            pv_cons = 1 - rien
        # print("Consistency test: %s p-value: %.5f" % (cons_test, pv_cons))
        Beta_hat = {'beta_hat': beta_hat[0], 'beta_hat_inf': beta_hat_inf[0], 'beta_hat_sup': beta_hat_sup[0]}
        return Beta_hat, "%.5f" % pv_cons

    def ribes(self, Cxi, Cy):
        """
               Aurelien Ribes model proposed in 2017 is implemented following the reference:
               Ribes, Aurelien, et al. (2017) A new statistical approach to climate change detection and attribution.
               Climate Dynamics. It considers the following set of equations:
                   Y_star = sum(X_star_i) for i from 1 to nf where nf is the number of forcings
                   Y = Y_star + epsilon_y
                   Xi = X_star_i + epsilon_xi
               Where epislon_y ~ N(0, Cy) and epislon_xi ~ N(0, Cxi)
               :param Cxi: numpy.ndarray
                   Covariance matrix for each of the forcings Xi. Should be a 3D array (nt, nt, nf)
               :param Cy: numpy.ndarray
                   Covariance matrix for the observations.
               :return:
               """
        X = self.X.sum(axis=1)
        Cx = Cxi.sum(axis=0)
        # Estimate the true state of variables (y) and (Xi) y_star and X_star_i using the MLE y_star_hat and
        # Xi_star_hat, respectively
        Xi_star_hat = np.zeros(self.X.shape)
        y_star_hat = self.y + np.linalg.multi_dot([Cy, np.linalg.inv(Cy + Cx), (X - self.y)])
        for i in range(Xi_star_hat.shape[1]):
            Xi_star_hat[:, i] = self.X[:, i] + np.linalg.multi_dot([Cxi[i], np.linalg.inv(Cy + Cx), (self.y - X)])
        # calculates variance for Y_star_hat
        Cy_star_hat = np.linalg.inv(np.linalg.inv(Cy) + np.linalg.inv(Cx))
        # calculates variance for Xi_star_hat
        Cxi_star_hat = np.zeros(Cxi.shape)
        for i in range(Cxi_star_hat.shape[0]):
            Cxi_temp = Cxi * 1.
            # sum for every j different than i
            Cxi_temp[i] = 0.
            Cxi_sum = Cxi_temp.sum(axis=0)
            Cxi_star_hat[i] = np.linalg.inv(np.linalg.inv(Cxi[i]) + np.linalg.inv(Cy + Cxi_sum))
        # hypothesis test: compare with chi-square distribution
        # print('#' * 60)
        print('Hypothesis testing p-value for Chi-2 distribution and Maximum Likelihood ...')
        # (internal variability only)
        d_cons = np.linalg.multi_dot([self.y.T, np.linalg.inv(Cy), self.y])
        print('%30s: %.7f (%.7f)' % ('Internal variability only', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        a0 = "%.7f" % chi2_test(d_cons, self.nt)
        # f.writelines('%30s: %.7f (%.7f)' % ('Internal variability only', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        # f.writelines("\n")
        # (all forcings)
        # d_cons = np.linalg.multi_dot([(self.y - X).T, np.linalg.inv(Cy + Cx), (self.y - X)])
        # print('%30s: %.7f (%.7f)' % ('All forcings', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        # f.writelines('%30s: %.7f (%.7f)' % ('All forcings', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        # f.writelines("\n")

        # a = "%.7f" % (np.exp(d_cons / -2.))
        # (individual forcings)
        # b0 = []
        # b = []
        # for i in range(self.X.shape[1]):
        #     d_cons = np.linalg.multi_dot([(self.y - self.X[:, i]).T, np.linalg.inv(Cy + Cxi[i]), (self.y - self.X[:, i])])
        #     print('%30s: %.7f (%.7f)' % ('Forcing no %d only' % (i + 1), chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        # f.writelines('%30s: %.7f (%.7f)' % ('Forcing no %d only' % (i + 1), chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))
        # f.writelines("\n")
        # b0.append("%.7f" % (chi2_test(d_cons, self.nt)))
        # b.append("%.7f" % (np.exp(d_cons / -2.)))
        return y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat, a0


########################                     Main                             ##############################
def main(y, X, Z, domain, jijie="ANN", zhishu="rx1day"):
    """
       Main method for using Ribes et al. (2017) algorithm including observational and model error
       :param y: numpy.ndarray
           Vector with nt observations
       :param X: numpy.ndarray
           nt x nf array with model data where nf is the number of forcings
       :param Z: numpy.ndarray
           nt x nz array of pseudo-observations used to calculate internal variability covariance matrix
       :param uncorr: numpy.ndarray
           nt x nz array of ensemble of observations representing uncorrelated error
       :param corr: numpy.ndarray
           nt x nz array of ensemble of observations representing correlated error
       :param init: int
           year to start analysis
       :param end: end
           year to end the analysis
       :return:
       """
    # preprocess - all
    p = PreProcess(y, X, Z)
    Z1, Z2 = p.extract_Z2(frac=0.5)
    yc, Xc, Z1c, Z2c = p.proj_fullrank(Z1, Z2)
    # Compute covariance matrices for internal variability
    Cv1 = p.creg(Z1c.T, method='ledoit')
    Cv2 = p.creg(Z2c.T, method='ledoit')
    # scale covariance matrix by number of ensemble members
    nt = len(y)
    Cx_ghg = Cm_estimate('historicalGHG', Cv2, X[:, 0], domain=domain, how_nr='pandas', jijie=jijie, zhishu=zhishu) + Cv_estimate('historicalGHG', Cv2, domain=domain, jijie=jijie, zhishu=zhishu, how_nr='pandas')
    Cx_nat = Cm_estimate('historicalNat', Cv2, X[:, 1], domain=domain, how_nr='pandas', jijie=jijie, zhishu=zhishu) + Cv_estimate('historicalNat', Cv2, domain=domain, jijie=jijie, zhishu=zhishu, how_nr='pandas')
    Cx_oa = Cm_estimate('historicalaer', Cv2, X[:, 2], domain=domain, how_nr='pandas', jijie=jijie, zhishu=zhishu) + Cv_estimate('historicalaer', Cv2, domain=domain, jijie=jijie, zhishu=zhishu, how_nr='pandas')
    Cx_hist = Cm_estimate('historical', Cv2, X[:, 3], domain=domain, how_nr='pandas', jijie=jijie, zhishu=zhishu) + Cv_estimate('historical', Cv2, domain=domain, jijie=jijie, zhishu=zhishu, how_nr='pandas')
    Cy = Cv1
    Cxi = np.stack([Cx_ghg, Cx_nat, Cx_oa, Cx_hist], axis=0)
    # starts attribution model
    m = AttributionModel(Xc, yc)
    y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat, a0 = m.ribes(Cxi, Cy)
    return y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat, a0


def get_trends_scaling_factors_Ribes_GPH():
    def _compute_slope(var):
        slp = linregress(range(len(var)), var).slope
        return slp

    trends_all = pd.DataFrame()
    for zhishu in ['SI']:
        for jijie in ['ANN']:
            for region in ["AUS", "EA", "IN", "land", "NAF", "NAM", "NHH", "NHM", "SAF", "SAM", "TRP", "WNP"]:
                # f.writelines(" ".join([zhishu, jijie, region]))
                # f.writelines("\n")
                df_y = pd.read_csv(dir + jijie + '/obs/' + jijie + '@obs@' + region + '@' + zhishu + '.csv', index_col=0)
                y = df_y['0'].values
                df_hist = pd.read_csv(dir + jijie + '/model/historical/multi_model/' + jijie + '@model@historical@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_X1_ghg = pd.read_csv(dir + jijie + '/model/historicalGHG/multi_model/' + jijie + '@model@historicalGHG@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_X2_nat = pd.read_csv(dir + jijie + '/model/historicalNat/multi_model/' + jijie + '@model@historicalNat@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_X3_aer = pd.read_csv(dir + jijie + '/model/historicalaer/multi_model/' + jijie + '@model@historicalaer@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                X = np.stack([df_X1_ghg, df_X2_nat, df_X3_aer, df_hist], axis=1)
                df_Z = pd.read_csv(dir + jijie + '/model/piControl/' + jijie + "@model@piControl@ensemble@" + region + '@' + zhishu + '.csv', index_col=0)
                Z = df_Z.values
                y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat, a0 = main(y, X, Z, domain=region, jijie=jijie, zhishu=zhishu)
                trends = all_trends(y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat)
                trends['zhishu'] = zhishu
                trends['jijie'] = jijie
                trends['domain'] = region
                trends['sf_best'] = np.NAN
                trends['sf_min'] = np.NAN
                trends['sf_max'] = np.NAN
                trends['IV_cons'] = a0
                trends.loc[0, 'sf_best'] = trends['trend'].loc[0] / _compute_slope(y)
                trends.loc[0, 'sf_min'] = trends['trend_min'].loc[0] / _compute_slope(y)
                trends.loc[0, 'sf_max'] = trends['trend_max'].loc[0] / _compute_slope(y)
                trends.loc[1, 'sf_best'] = trends['trend'].loc[1] / _compute_slope(df_X1_ghg)
                trends.loc[1, 'sf_min'] = trends['trend_min'].loc[1] / _compute_slope(df_X1_ghg)
                trends.loc[1, 'sf_max'] = trends['trend_max'].loc[1] / _compute_slope(df_X1_ghg)
                trends.loc[2, 'sf_best'] = trends['trend'].loc[2] / _compute_slope(df_X2_nat)
                trends.loc[2, 'sf_min'] = trends['trend_min'].loc[2] / _compute_slope(df_X2_nat)
                trends.loc[2, 'sf_max'] = trends['trend_max'].loc[2] / _compute_slope(df_X2_nat)
                trends.loc[3, 'sf_best'] = trends['trend'].loc[3] / _compute_slope(df_X3_aer)
                trends.loc[3, 'sf_min'] = trends['trend_min'].loc[3] / _compute_slope(df_X3_aer)
                trends.loc[3, 'sf_max'] = trends['trend_max'].loc[3] / _compute_slope(df_X3_aer)
                trends.loc[4, 'sf_best'] = trends['trend'].loc[4] / _compute_slope(df_hist)
                trends.loc[4, 'sf_min'] = trends['trend_min'].loc[4] / _compute_slope(df_hist)
                trends.loc[4, 'sf_max'] = trends['trend_max'].loc[4] / _compute_slope(df_hist)
                trends_all = pd.concat([trends_all, trends], axis=0, ignore_index=True)
    trends_all.to_csv(dir + 'trends_scaling_factors_Ribes_GPH.csv')


def get_trends_scaling_factors_2signal_GPH():
    def _compute_slope(var):
        slp = linregress(range(len(var)), var).slope
        return slp

    trends_all = pd.DataFrame()
    for zhishu in ["SI"]:
        for jijie in ['ANN']:
            for region in ["AUS", "EA", "IN", "land", "NAF", "NAM", "NHH", "NHM", "SAF", "SAM", "TRP", "WNP"]:
                df_y = pd.read_csv(dir + jijie + '/obs/' + jijie + '@obs@' + region + '@' + zhishu + '.csv', index_col=0)
                y = df_y['0'].values
                df_hist = pd.read_csv(dir + jijie + '/model/historical/multi_model/' + jijie + '@model@historical@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_nat = pd.read_csv(dir + jijie + '/model/historicalNat/multi_model/' + jijie + '@model@historicalNat@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_ant = df_hist - df_nat
                X = np.stack([df_ant, df_nat], axis=1)
                df_Z = pd.read_csv(dir + jijie + '/model/piControl/' + jijie + "@model@piControl@ensemble@" + region + '@' + zhishu + '.csv', index_col=0)
                Z = df_Z.values
                p = PreProcess(y, X, Z)
                Z1, Z2 = p.extract_Z2(frac=0.45)
                yc, Xc, Z1c, Z2c = p.proj_fullrank(Z1, Z2)
                Cr = p.creg(Z1c.T, method='ledoit')
                m = AttributionModel(Xc, yc)
                beta_ols, pv_cons = m.ols(Cr, np.array([[1, 0], [0, 1]]), Z2c)
                # print(zhishu, jijie, region, pv_cons)
                beta_ols = pd.DataFrame(beta_ols)
                beta_ols.columns = ['sf_best', 'sf_min', 'sf_max']
                beta_ols['trend'] = np.NAN
                beta_ols['trend_min'] = np.NAN
                beta_ols['trend_max'] = np.NAN
                beta_ols.loc[0, 'trend'] = beta_ols.loc[0, 'sf_best'] * _compute_slope(df_ant)
                beta_ols.loc[1, 'trend'] = beta_ols.loc[1, 'sf_best'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend'] = beta_ols.loc[0, 'trend'] + beta_ols.loc[1, 'trend']
                beta_ols.loc[0, 'trend_min'] = beta_ols.loc[0, 'sf_min'] * _compute_slope(df_ant)
                beta_ols.loc[1, 'trend_min'] = beta_ols.loc[1, 'sf_min'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend_min'] = beta_ols.loc[0, 'trend_min'] + beta_ols.loc[1, 'trend_min']
                beta_ols.loc[0, 'trend_max'] = beta_ols.loc[0, 'sf_max'] * _compute_slope(df_ant)
                beta_ols.loc[1, 'trend_max'] = beta_ols.loc[1, 'sf_max'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend_max'] = beta_ols.loc[0, 'trend_max'] + beta_ols.loc[1, 'trend_max']
                # beta_ols.index = ['ANT','NAT','ALL']
                beta_ols['forcing'] = ['ANT', 'NAT', 'ALL']
                beta_ols['domain'] = region
                beta_ols['zhishu'] = zhishu
                beta_ols['jijie'] = jijie
                beta_ols['pv_cons'] = pv_cons
                trends_all = pd.concat([trends_all, beta_ols], axis=0, ignore_index=True)
    trends_all.to_csv(dir + 'trends_scaling_factors_2signal_GPH.csv')


def get_trends_scaling_factors_3signal_GPH():
    def _compute_slope(var):
        slp = linregress(range(len(var)), var).slope
        return slp

    trends_all = pd.DataFrame()
    for zhishu in ['SI']:
        for jijie in ['ANN']:
            for region in ["AUS", "EA", "IN", "land", "NAF", "NAM", "NHH", "NHM", "SAF", "SAM", "TRP", "WNP"]:
                df_y = pd.read_csv(dir + jijie + '/obs/' + jijie + '@obs@' + region + '@' + zhishu + '.csv', index_col=0)
                y = df_y['0'].values
                df_hist = pd.read_csv(dir + jijie + '/model/historical/multi_model/' + jijie + '@model@historical@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_ghg = pd.read_csv(dir + jijie + '/model/historicalGHG/multi_model/' + jijie + '@model@historicalGHG@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_nat = pd.read_csv(dir + jijie + '/model/historicalNat/multi_model/' + jijie + '@model@historicalNat@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                df_aer = pd.read_csv(dir + jijie + '/model/historicalaer/multi_model/' + jijie + '@model@historicalaer@multi_model@' + region + '@' + zhishu + '@modelMean.csv')['0'].values
                # df_aer = df_hist - df_ghg - df_nat
                X = np.stack([df_ghg, df_nat, df_aer], axis=1)
                df_Z = pd.read_csv(dir + jijie + '/model/piControl/' + jijie + "@model@piControl@ensemble@" + region + '@' + zhishu + '.csv', index_col=0)
                Z = df_Z.values
                p = PreProcess(y, X, Z)
                Z1, Z2 = p.extract_Z2(frac=0.45)
                yc, Xc, Z1c, Z2c = p.proj_fullrank(Z1, Z2)
                Cr = p.creg(Z1c.T, method='ledoit')
                m = AttributionModel(Xc, yc)
                beta_ols, pv_cons = m.ols(Cr, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), Z2c)
                # print(zhishu, jijie, region, pv_cons)
                beta_ols = pd.DataFrame(beta_ols)
                beta_ols.columns = ['sf_best', 'sf_min', 'sf_max']
                beta_ols['trend'] = np.NAN
                beta_ols['trend_min'] = np.NAN
                beta_ols['trend_max'] = np.NAN
                beta_ols.loc[0, 'trend'] = beta_ols.loc[0, 'sf_best'] * _compute_slope(df_ghg)
                beta_ols.loc[1, 'trend'] = beta_ols.loc[1, 'sf_best'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend'] = beta_ols.loc[2, 'sf_best'] * _compute_slope(df_aer)
                beta_ols.loc[3, 'trend'] = beta_ols.loc[0, 'trend'] + beta_ols.loc[1, 'trend'] + beta_ols.loc[2, 'trend']
                beta_ols.loc[0, 'trend_min'] = beta_ols.loc[0, 'sf_min'] * _compute_slope(df_ghg)
                beta_ols.loc[1, 'trend_min'] = beta_ols.loc[1, 'sf_min'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend_min'] = beta_ols.loc[2, 'sf_min'] * _compute_slope(df_aer)
                beta_ols.loc[3, 'trend_min'] = beta_ols.loc[0, 'trend_min'] + beta_ols.loc[1, 'trend_min'] + beta_ols.loc[2, 'trend_min']
                beta_ols.loc[0, 'trend_max'] = beta_ols.loc[0, 'sf_max'] * _compute_slope(df_ghg)
                beta_ols.loc[1, 'trend_max'] = beta_ols.loc[1, 'sf_max'] * _compute_slope(df_nat)
                beta_ols.loc[2, 'trend_max'] = beta_ols.loc[2, 'sf_max'] * _compute_slope(df_aer)
                beta_ols.loc[3, 'trend_max'] = beta_ols.loc[0, 'trend_max'] + beta_ols.loc[1, 'trend_max'] + beta_ols.loc[2, 'trend_max']
                beta_ols.loc[3, 'sf_best'] = beta_ols.loc[3, 'trend'] / _compute_slope(df_y['0'])
                beta_ols.loc[3, 'sf_min'] = beta_ols.loc[3, 'trend_min'] / _compute_slope(df_y['0'])
                beta_ols.loc[3, 'sf_max'] = beta_ols.loc[3, 'trend_max'] / _compute_slope(df_y['0'])
                # beta_ols['forcing'] = ['GHG','NAT','AER']
                beta_ols['forcing'] = ['GHG', 'NAT', 'AER', 'ALL']
                beta_ols['domain'] = region
                beta_ols['zhishu'] = zhishu
                beta_ols['jijie'] = jijie
                beta_ols['pv_cons'] = pv_cons
                trends_all = pd.concat([trends_all, beta_ols], axis=0, ignore_index=True)
    trends_all.to_csv(dir + 'trends_scaling_factors_3signal_GPH.csv')


if __name__ == '__main__':
    get_trends_scaling_factors_Ribes_GPH()
    get_trends_scaling_factors_2signal_GPH()
    get_trends_scaling_factors_3signal_GPH()
