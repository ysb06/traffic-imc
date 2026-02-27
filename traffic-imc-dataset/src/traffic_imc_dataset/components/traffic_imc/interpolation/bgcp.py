import pandas as pd
import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from tqdm import tqdm
from .base import Interpolator


class BGCPInterpolator(Interpolator):
    """
    Missing-value interpolator based on
    Bayesian Gaussian CP decomposition (BGCP).
    
    Reference:
        Xinyu Chen, Zhaocheng He, Lijun Sun (2019). 
        A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation.
        Transportation Research Part C: Emerging Technologies, 98: 73-84.
    
    Input DataFrame format:
        - index: DatetimeIndex (hourly, 1-hour interval)
        - columns: sensor IDs
        - values: traffic volume (float), with NaN for missing values

    Tensor shape: (sensors m × days n × hour-of-day f=24)
    """
    
    def __init__(
        self, 
        rank: int = 30,
        burn_iter: int = 200,
        gibbs_iter: int = 100,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        """
        Args:
            rank: CP decomposition rank (number of latent factors)
            burn_iter: Number of burn-in iterations
            gibbs_iter: Number of Gibbs sampling iterations
            random_seed: Random seed for reproducibility
            fallback_method: Fallback method for remaining NaNs after BGCP
                           ('linear', 'ffill', 'bfill', 'median')
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
        """
        self.name = self.__class__.__name__
        self.rank = rank
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose
    
    # ==================== Core BGCP helpers ====================
    
    @staticmethod
    def _mvnrnd_pre(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Sample from multivariate normal using a precision matrix."""
        src = normrnd(size=(mu.shape[0],))
        return solve_ut(
            cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
            src, lower=False, check_finite=False, overwrite_b=True
        ) + mu
    
    @staticmethod
    def _cp_combine(factor: list[np.ndarray]) -> np.ndarray:
        """Reconstruct tensor from CP factors."""
        return np.einsum('is, js, ts -> ijt', factor[0], factor[1], factor[2])
    
    @staticmethod
    def _ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold tensor into matrix form."""
        return np.reshape(
            np.moveaxis(tensor, mode, 0), 
            (tensor.shape[mode], -1), 
            order='F'
        )
    
    @staticmethod
    def _cov_mat(mat: np.ndarray, mat_bar: np.ndarray) -> np.ndarray:
        """Compute covariance matrix."""
        mat = mat - mat_bar
        return mat.T @ mat
    
    def _sample_factor(
        self,
        tau_sparse_tensor: np.ndarray,
        tau_ind: np.ndarray,
        factor: list[np.ndarray],
        k: int,
        beta0: float = 1.0
    ) -> np.ndarray:
        """Sample latent factors."""
        dim, rank = factor[k].shape
        factor_bar = np.mean(factor[k], axis=0)
        temp = dim / (dim + beta0)
        var_mu_hyper = temp * factor_bar
        var_W_hyper = inv(
            np.eye(rank) + 
            self._cov_mat(factor[k], factor_bar) + 
            temp * beta0 * np.outer(factor_bar, factor_bar)
        )
        var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
        var_mu_hyper = self._mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)
        
        idx = list(filter(lambda x: x != k, range(len(factor))))
        var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self._ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
        var4 = var1 @ self._ten2mat(tau_sparse_tensor, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
        
        for i in range(dim):
            factor[k][i, :] = self._mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
        return factor[k]
    
    @staticmethod
    def _sample_precision_tau(
        sparse_tensor: np.ndarray,
        tensor_hat: np.ndarray,
        ind: np.ndarray
    ) -> float:
        """Sample precision parameter tau."""
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)
    
    # ==================== Main BGCP algorithm ====================
    
    def _bgcp_impute(self, sparse_tensor: np.ndarray) -> np.ndarray:
        """
        Restore missing values using the BGCP algorithm.

        Args:
            sparse_tensor: 3D tensor with NaN missing entries (sensor × day × hour)

        Returns:
            Reconstructed tensor
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        dim = np.array(sparse_tensor.shape)
        rank = self.rank
        
        # Identify missing positions (NaN handling)
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor_filled = sparse_tensor.copy()
        sparse_tensor_filled[np.isnan(sparse_tensor_filled)] = 0

        # Initialize factors
        factor = [0.1 * np.random.randn(dim[k], rank) for k in range(len(dim))]
        
        tau = 1.0
        factor_plus = [np.zeros((dim[k], rank)) for k in range(len(dim))]
        tensor_hat_plus = np.zeros(dim)
        
        total_iter = self.burn_iter + self.gibbs_iter
        
        # Configure tqdm progress bar
        iterator = tqdm(
            range(total_iter),
            desc="BGCP Sampling",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        
        for it in iterator:
            tau_ind = tau * ind
            tau_sparse_tensor = tau * sparse_tensor_filled

            # Sample factors for each mode
            for k in range(len(dim)):
                factor[k] = self._sample_factor(tau_sparse_tensor, tau_ind, factor, k)
            
            tensor_hat = self._cp_combine(factor)
            tau = self._sample_precision_tau(sparse_tensor_filled, tensor_hat, ind)
            
            # Accumulate samples after burn-in
            if it + 1 > self.burn_iter:
                factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
                tensor_hat_plus += tensor_hat

            # Update tqdm status
            phase = "Burn-in" if it < self.burn_iter else "Gibbs"
            iterator.set_postfix(phase=phase, tau=f"{tau:.4f}")

        # Average Gibbs samples
        tensor_hat = tensor_hat_plus / self.gibbs_iter
        
        return tensor_hat
    
    # ==================== Data reshape helpers ====================
    
    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        Convert DataFrame into a BGCP-compatible 3D tensor.

        Tensor shape: (sensors m × days n × hour-of-day f=24)

        Args:
            df: Input DataFrame (index: datetime, columns: sensor IDs)

        Returns:
            tensor: 3D numpy array (m × n × 24)
            meta: Metadata used for inverse reshape
        """
        # Number of hours per day (24 for hourly data)
        hours_per_day = 24

        # Sensor count and total number of hours
        num_sensors = len(df.columns)
        num_hours = len(df)

        # Compute day count (exclude incomplete trailing day)
        num_days = num_hours // hours_per_day
        valid_hours = num_days * hours_per_day

        # Remove incomplete final day
        df_trimmed = df.iloc[:valid_hours]

        # DataFrame -> 2D array (time × sensor) -> transpose (sensor × time)
        mat = df_trimmed.values.T  # (sensor × time)

        # Reshape to 3D tensor: (sensor m × day n × hour f=24)
        tensor = mat.reshape(num_sensors, num_days, hours_per_day)

        # Save metadata for inverse reshape
        meta = {
            'columns': df.columns,
            'index': df_trimmed.index,
            'original_index': df.index,
            'original_length': len(df),
            'trimmed_length': valid_hours,
            'num_sensors': num_sensors,
            'num_days': num_days,
        }
        
        return tensor, meta
    
    def _inverse_reshape(
        self, 
        tensor: np.ndarray, 
        meta: dict
    ) -> pd.DataFrame:
        """
        Restore BGCP output tensor to original DataFrame layout.

        Args:
            tensor: Reconstructed 3D tensor (sensor × day × hour)
            meta: Metadata from reshape

        Returns:
            Restored DataFrame
        """
        num_sensors = meta['num_sensors']
        num_days = meta['num_days']
        hours_per_day = 24
        
        # 3D -> 2D: (sensor × day × 24) -> (sensor × time)
        mat = tensor.reshape(num_sensors, num_days * hours_per_day)

        # Transpose: (sensor × time) -> (time × sensor)
        mat = mat.T

        # Build DataFrame
        df_result = pd.DataFrame(
            mat,
            index=meta['index'],
            columns=meta['columns']
        )
        
        return df_result
    
    # ==================== Main interpolation flow ====================
    
    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fallback handling for NaNs remaining after BGCP."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after BGCP. "
                f"Applying fallback method: {self.fallback_method}"
            )

        fallback_strategies = {
            "linear": lambda x: x.interpolate(
                method="linear", limit_direction="both", axis=0
            ),
            "ffill": lambda x: x.ffill().bfill(),
            "bfill": lambda x: x.bfill().ffill(),
            "median": lambda x: x.fillna(x.median()),
        }

        df = fallback_strategies.get(
            self.fallback_method, fallback_strategies["linear"]
        )(df)

        # Final safety net
        final_nans = df.isna().sum().sum()
        if final_nans > 0:
            if self.verbose > 0:
                print(
                    f"Warning: {final_nans} NaN values still remain. "
                    f"Filling with 0 as last resort."
                )
            df = df.fillna(0)

        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run missing-value interpolation with BGCP.

        Args:
            df: Input DataFrame (NaN indicates missing values)

        Returns:
            DataFrame with imputed missing values
        """
        original_length = len(df)

        # 1. DataFrame -> 3D tensor
        sparse_tensor, meta = self._reshape(df)
        
        if self.verbose > 0:
            print(f"Tensor shape: {sparse_tensor.shape} "
                  f"(sensors={meta['num_sensors']}, days={meta['num_days']}, hours=24)")
            nan_count = np.isnan(sparse_tensor).sum()
            total_count = sparse_tensor.size
            print(f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)")
            if meta['trimmed_length'] < original_length:
                print(f"Note: Last {original_length - meta['trimmed_length']} hours "
                      f"trimmed (not divisible by 24)")
        
        # 2. Reconstruct missing values with BGCP
        if self.verbose > 0:
            print(f"Running BGCP (rank={self.rank}, burn_iter={self.burn_iter}, "
                  f"gibbs_iter={self.gibbs_iter})...")
        
        imputed_tensor = self._bgcp_impute(sparse_tensor)
        
        # 3. 3D tensor -> DataFrame
        df_imputed = self._inverse_reshape(imputed_tensor, meta)

        # 4. Replace only originally missing entries
        df_result = df.copy()
        trimmed_index = meta['index']
        mask = df.loc[trimmed_index].isna()
        df_result.loc[trimmed_index] = df_result.loc[trimmed_index].where(~mask, df_imputed)

        # 5. Apply fallback (trimmed tail + remaining missing values)
        df_result = self._apply_fallback(df_result)
        
        if self.verbose > 0:
            print("BGCP interpolation completed.")
        
        return df_result
