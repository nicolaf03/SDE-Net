import pandas as pd
import numpy as np


class FractionalBM:
    def __init__(self):
        self.s0 = 0
        self.theta = 0
        self.sigma = 0
        self.hurst = 0
        #ToDO: let s consider to inherit the seasonality funct as a class

    def seasonality_func(self, t):
        raise Exception('Not implemented yet')

    def set_seasonality_pars(self, params):
        raise Exception('Not implemented yet')

    def set_parameters(self, params):
        """
        :param params: vector of parameters defining the class FractionalBM
        :return:
        """
        assert len(params) == 5, 'Number of parameters is {}. Expected: 4'.format(len(params))
        self.s0 = params[0]
        self.mu = params[1]
        self.theta = params[2]
        self.sigma = params[3]
        self.hurst = params[4]  # not the h-index for publications (:

    def _autocovariance(self, s, t):
        """Autocovariance for fractional_BM given a distance."""
        return 0.5 * (s ** (2 * self.hurst) + t ** (2 * self.hurst) - abs(t - s) ** (2 * self.hurst))

    def _autocovariance_mat(self, time_vec):
        """Autocovariance Squared Matrix for fractional_BM given a time vector."""
        n_t = len(time_vec)
        gamma_mat = np.empty([n_t, n_t])
        for s_idx, s in enumerate(time_vec):
            for t_idx, t in enumerate(time_vec):
                gamma_mat[s_idx, t_idx] = self._autocovariance(s, t)
        return gamma_mat

    def simulate(self, n_sims: int, t_steps: int, dt: float) -> pd.DataFrame:
        """
        Implements the Eq. 12 of Stochastic Modeling of Wind Derivatives in Energy Markets, by Benth, Di Persio, Lavagnini. (2018)
        :param n_sims: number of simulations
        :param t_steps: number of future time steps
        :param dt: temporal increment (assumed to be constant)
        :return: a dataframe containing the n_sims x t_steps trajectories
        """
        #Cholesky's method (not the most efficient), see https://github.com/732jhy/Fractional-Brownian-Motion
        # See also https://en.wikipedia.org/wiki/Fractional_Brownian_motion method1

        # gamma = lambda s, t: self._autocovariance(s, t)
        simulations = np.arange(0, n_sims)
        time_span = np.arange(0, (t_steps + 1) * dt, dt)
        simulation_df = pd.DataFrame(data=np.empty((n_sims, t_steps + 1)), index=simulations, columns=time_span)
        simulation_df.loc[:, 0] = self.s0

        gamma_mat = self._autocovariance_mat(time_span[1:])
        sigma_mat = np.linalg.cholesky(gamma_mat)

        v_mat = np.random.randn(t_steps, n_sims) # Increment of a standard BM \sim (0, 1)
        fbm = np.matmul(sigma_mat, v_mat) # t_steps x n_sims

        for n_sim in simulations:
            fbm_tmp = fbm[:, n_sim]
            # we assume the discretization of the integral is on base dt

            simulation_df.loc[n_sim, dt:] = self.s0 * np.exp(-self.theta * time_span[1:]) +\
                                            self.mu * (1 - np.exp(-self.theta * time_span[1:])) \
                                            + self.sigma * np.exp(-self.theta * time_span[1:]) * fbm_tmp

            # simulation_df.loc[n_sim, dt:] = self.s0 + self.mu * time_span[1:] + np.cumsum(v_mat[:, n_sim]) * self.sigma * np.sqrt(time_span[1:])
        return simulation_df