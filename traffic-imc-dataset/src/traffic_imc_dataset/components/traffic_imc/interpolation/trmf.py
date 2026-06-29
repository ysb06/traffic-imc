import pandas as pd
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
from .base import Interpolator


class TRMFInterpolator(Interpolator):
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

    def _update_W(
        self,
        sparse_mat: np.ndarray,
        binary_mat: np.ndarray,
        X: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        dim1 = sparse_mat.shape[0]
        rank = self.rank
        
        for i in range(dim1):
            pos0 = np.where(binary_mat[i, :] == 1)[0]  # Use binary observation mask
            if len(pos0) == 0:
                continue
            Xt = X[pos0, :]
            vec0 = Xt.T @ sparse_mat[i, pos0]
            mat0 = Xt.T @ Xt + self.lambda_w * np.eye(rank)
            W[i, :] = np.linalg.solve(mat0, vec0)
        
        return W
    
    def _update_X(
        self,
        sparse_mat: np.ndarray,
        binary_mat: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        dim2 = sparse_mat.shape[1]
        rank = self.rank
        time_lags = self.time_lags
        d = len(time_lags)
        
        for t in range(dim2):
            pos0 = np.where(binary_mat[:, t] == 1)[0]
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
            X[t, :] = np.linalg.solve(mat0, vec0)
        
        return X
    
    def _update_theta(
        self,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
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
            W = self._update_W(sparse_mat, binary_mat, X, W)
            X = self._update_X(sparse_mat, binary_mat, W, X, theta)
            theta = self._update_theta(X, theta)

            if (it + 1) % 10 == 0:
                mat_hat = W @ X.T
                pos_obs = np.where(binary_mat == 1)
                if len(pos_obs[0]) > 0:
                    rmse = np.sqrt(np.mean((sparse_mat[pos_obs] - mat_hat[pos_obs]) ** 2))
                    iterator.set_postfix(rmse=f"{rmse:.4f}")
        
        mat_hat = W @ X.T
        
        return mat_hat
    
    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
        mat = df.values.T.copy()
        binary_mat = (~np.isnan(mat)).astype(np.float64)
        mat = np.nan_to_num(mat, nan=0.0)
        
        meta = {
            'columns': df.columns,
            'index': df.index,
            'num_sensors': len(df.columns),
            'num_times': len(df),
        }
        
        return mat, binary_mat, meta
    
    def _inverse_reshape(self, mat: np.ndarray, meta: dict) -> pd.DataFrame:
        values = mat.T
        df_result = pd.DataFrame(
            values,
            index=meta['index'],
            columns=meta['columns']
        )
        
        return df_result
    
    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
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
        sparse_mat, binary_mat, meta = self._reshape(df)
        
        if self.verbose > 0:
            print(f"Matrix shape: {sparse_mat.shape} "
                  f"(sensors={meta['num_sensors']}, times={meta['num_times']})")
            nan_count = (binary_mat == 0).sum()
            total_count = sparse_mat.size
            print(f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)")
            print(f"Running TRMF (rank={self.rank}, time_lags={self.time_lags.tolist()}, "
                  f"maxiter={self.maxiter})...")
        
        imputed_mat = self._trmf_impute(sparse_mat, binary_mat)
        df_imputed = self._inverse_reshape(imputed_mat, meta)
        df_result = df.copy()
        nan_mask_df = df.isna()
        df_result[nan_mask_df] = df_imputed[nan_mask_df]
        df_result = self._apply_fallback(df_result)
        
        if self.verbose > 0:
            print("TRMF interpolation completed.")
        
        return df_result
