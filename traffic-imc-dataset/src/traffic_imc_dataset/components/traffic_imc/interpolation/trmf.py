import pandas as pd
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
from .base import Interpolator


class TRMFInterpolator(Interpolator):
    """
    Missing-value interpolator based on
    Temporal Regularized Matrix Factorization (TRMF).
    
    Reference:
        Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon (2016).
        Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction.
        30th Conference on Neural Information Processing Systems (NIPS 2016).
    
    Input DataFrame format:
        - index: DatetimeIndex (hourly)
        - columns: sensor IDs
        - values: traffic volume (float), with NaN for missing values

    Matrix shape: (sensors m × timesteps f)
    """
    
    def __init__(
        self,
        rank: int = 20,
        time_lags: list[int] | None = None,
        maxiter: int = 200,
        lambda_w: float = 500.0,
        lambda_x: float = 500.0,
        lambda_theta: float = 500.0,
        eta: float = 0.03,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        """
        Args:
            rank: Factorization rank (number of latent factors)
            time_lags: AR model time lag list (default: [1, 2, 24])
            maxiter: Maximum number of iterations
            lambda_w: Regularization coefficient for spatial matrix W
            lambda_x: Regularization coefficient for temporal matrix X
            lambda_theta: Regularization coefficient for AR coefficients theta
            eta: L2 regularization coefficient for X
            random_seed: Random seed for reproducibility
            fallback_method: Fallback method for remaining NaNs after TRMF
                           ('linear', 'ffill', 'bfill', 'median')
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
        """
        self.name = self.__class__.__name__
        self.rank = rank
        self.time_lags = np.array(time_lags) if time_lags is not None else np.array([1, 2, 24])
        self.maxiter = maxiter
        self.lambda_w = lambda_w
        self.lambda_x = lambda_x
        self.lambda_theta = lambda_theta
        self.eta = eta
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose
    
    # ==================== Core TRMF algorithm ====================
    
    def _update_W(
        self,
        sparse_mat: np.ndarray,
        binary_mat: np.ndarray,
        X: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """Update spatial matrix W."""
        dim1 = sparse_mat.shape[0]
        rank = self.rank
        
        for i in range(dim1):
            pos0 = np.where(binary_mat[i, :] == 1)[0]  # Use binary observation mask
            if len(pos0) == 0:
                continue
            Xt = X[pos0, :]
            vec0 = Xt.T @ sparse_mat[i, pos0]
            mat0 = Xt.T @ Xt + self.lambda_w * np.eye(rank)
            W[i, :] = np.linalg.solve(mat0, vec0)  # Prefer solve over matrix inverse
        
        return W
    
    def _update_X(
        self,
        sparse_mat: np.ndarray,
        binary_mat: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Update temporal matrix X with AR constraints."""
        dim2 = sparse_mat.shape[1]
        rank = self.rank
        time_lags = self.time_lags
        d = len(time_lags)
        
        for t in range(dim2):
            pos0 = np.where(binary_mat[:, t] == 1)[0]  # Use binary observation mask
            if len(pos0) == 0:
                Wt = np.zeros((1, rank))
            else:
                Wt = W[pos0, :]
            
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            
            if t < np.max(time_lags):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2))[0])
                
                for k in index:
                    Ak = theta[k, :]
                    Mt += np.diag(Ak ** 2)
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Nt += np.multiply(
                        Ak,
                        X[t + time_lags[k], :] - np.einsum('ij, ij -> j', theta0, X[t + time_lags[k] - time_lags, :])
                    )
            
            if len(pos0) == 0:
                vec0 = self.lambda_x * Nt + self.lambda_x * Qt
            else:
                vec0 = Wt.T @ sparse_mat[pos0, t] + self.lambda_x * Nt + self.lambda_x * Qt
            
            mat0 = Wt.T @ Wt + self.lambda_x * Mt + self.lambda_x * Pt + self.lambda_x * self.eta * np.eye(rank)
            X[t, :] = np.linalg.solve(mat0, vec0)  # Prefer solve over matrix inverse
        
        return X
    
    def _update_theta(
        self,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Update AR coefficients theta."""
        dim2 = X.shape[0]
        rank = self.rank
        time_lags = self.time_lags
        d = len(time_lags)
        
        for k in range(d):
            theta0 = theta.copy()
            theta0[k, :] = 0
            mat0 = np.zeros((dim2 - np.max(time_lags), rank))
            
            for L in range(d):
                mat0 += X[np.max(time_lags) - time_lags[L] : dim2 - time_lags[L], :] @ np.diag(theta0[L, :])
            
            VarPi = X[np.max(time_lags) : dim2, :] - mat0
            var1 = np.zeros((rank, rank))
            var2 = np.zeros(rank)
            
            for t in range(np.max(time_lags), dim2):
                B = X[t - time_lags[k], :]
                var1 += np.diag(np.multiply(B, B))
                var2 += np.diag(B) @ VarPi[t - np.max(time_lags), :]
            
            mat0 = var1 + self.lambda_theta * np.eye(rank) / self.lambda_x
            theta[k, :] = np.linalg.solve(mat0, var2)  # Prefer solve over matrix inverse
        
        return theta
    
    def _trmf_impute(self, sparse_mat: np.ndarray, binary_mat: np.ndarray) -> np.ndarray:
        """
        Restore missing values with the TRMF algorithm.

        Args:
            sparse_mat: Observation matrix (sensor × time), missing entries can be placeholder values
            binary_mat: Observation mask (sensor × time), 1=observed, 0=missing

        Returns:
            Reconstructed matrix
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        dim1, dim2 = sparse_mat.shape
        rank = self.rank
        time_lags = self.time_lags
        d = len(time_lags)
        
        # Initialize parameters
        W = 0.1 * np.random.rand(dim1, rank)
        X = 0.1 * np.random.rand(dim2, rank)
        theta = 0.1 * np.random.rand(d, rank)
        
        # Configure tqdm progress bar
        iterator = tqdm(
            range(self.maxiter),
            desc="TRMF Optimization",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        
        for it in iterator:
            # 1. Update W
            W = self._update_W(sparse_mat, binary_mat, X, W)

            # 2. Update X
            X = self._update_X(sparse_mat, binary_mat, W, X, theta)

            # 3. Update theta
            theta = self._update_theta(X, theta)

            # Update tqdm status
            if (it + 1) % 10 == 0:
                mat_hat = W @ X.T
                # Compute RMSE on observed entries
                pos_obs = np.where(binary_mat == 1)  # Use binary observation mask
                if len(pos_obs[0]) > 0:
                    rmse = np.sqrt(np.mean((sparse_mat[pos_obs] - mat_hat[pos_obs]) ** 2))
                    iterator.set_postfix(rmse=f"{rmse:.4f}")
        
        # Final reconstructed matrix
        mat_hat = W @ X.T
        
        return mat_hat
    
    # ==================== Data reshape helpers ====================
    
    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Convert DataFrame to a TRMF-compatible 2D matrix.

        Matrix shape: (sensors m × timesteps f)

        Args:
            df: Input DataFrame (index: datetime, columns: sensor IDs)

        Returns:
            mat: 2D numpy array (sensor × time), with NaN temporarily filled by 0
            binary_mat: Observation mask (sensor × time), 1=observed, 0=missing
            meta: Metadata used for inverse reshape
        """
        # DataFrame -> (time × sensor) -> transpose -> (sensor × time)
        mat = df.values.T.copy()

        # Build binary mask: 1=observed, 0=missing
        binary_mat = (~np.isnan(mat)).astype(np.float64)

        # Temporarily replace NaN with 0 (mask keeps missingness information)
        mat = np.nan_to_num(mat, nan=0.0)
        
        meta = {
            'columns': df.columns,
            'index': df.index,
            'num_sensors': len(df.columns),
            'num_times': len(df),
        }
        
        return mat, binary_mat, meta
    
    def _inverse_reshape(self, mat: np.ndarray, meta: dict) -> pd.DataFrame:
        """
        Restore TRMF matrix output to original DataFrame layout.

        Args:
            mat: Reconstructed 2D matrix (sensor × time)
            meta: Metadata from reshape

        Returns:
            Restored DataFrame
        """
        # (sensor × time) -> transpose -> (time × sensor)
        values = mat.T

        # Build DataFrame
        df_result = pd.DataFrame(
            values,
            index=meta['index'],
            columns=meta['columns']
        )
        
        return df_result
    
    # ==================== Main interpolation flow ====================
    
    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fallback handling for NaNs remaining after TRMF."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after TRMF. "
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
        Run missing-value interpolation with TRMF.

        Args:
            df: Input DataFrame (NaN indicates missing values)

        Returns:
            DataFrame with imputed missing values
        """
        # 1. DataFrame -> 2D matrix
        sparse_mat, binary_mat, meta = self._reshape(df)
        
        if self.verbose > 0:
            print(f"Matrix shape: {sparse_mat.shape} "
                  f"(sensors={meta['num_sensors']}, times={meta['num_times']})")
            nan_count = (binary_mat == 0).sum()
            total_count = sparse_mat.size
            print(f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)")
            print(f"Running TRMF (rank={self.rank}, time_lags={self.time_lags.tolist()}, "
                  f"maxiter={self.maxiter})...")
        
        # 2. Reconstruct missing values with TRMF
        imputed_mat = self._trmf_impute(sparse_mat, binary_mat)

        # 3. 2D matrix -> DataFrame
        df_imputed = self._inverse_reshape(imputed_mat, meta)

        # 4. Replace only originally missing entries
        df_result = df.copy()
        nan_mask_df = df.isna()
        df_result[nan_mask_df] = df_imputed[nan_mask_df]

        # 5. Apply fallback for any remaining missing values
        df_result = self._apply_fallback(df_result)
        
        if self.verbose > 0:
            print("TRMF interpolation completed.")
        
        return df_result
