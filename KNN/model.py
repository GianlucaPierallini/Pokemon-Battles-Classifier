"""Sparse-aware kNN preprocessing, CPU / optional CuPy CUDA neighbor lookup."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.sparsefuncs import inplace_csr_column_scale

from .features import BattleFeatureExtractor

LOG = logging.getLogger(__name__)


class SparseFeatureScaler:
    """Scale features learned on a training fold, keeping the matrix sparse.

    'all':     original notebook StandardScaler(with_mean=False), INCLUDING rare one-hots
    'numeric': numeric columns standardized; one-hot columns unchanged (recommended)
    'none':    leave every feature untouched
    """

    def __init__(self, mode: str = "numeric", category_weight: float = 1.0) -> None:
        if mode not in ("all", "numeric", "none"):
            raise ValueError("scaling must be all, numeric or none")
        if category_weight <= 0:
            raise ValueError("category_weight must be positive")
        self.mode = mode
        self.category_weight = category_weight
        self.factors_: np.ndarray | None = None

    def fit(self, X: sparse.csr_matrix, feature_names: list[str]):
        X = X.tocsr()
        if self.mode == "none":
            scales = np.ones(X.shape[1], dtype=np.float64)
        else:
            fitted = StandardScaler(with_mean=False).fit(X)
            scales = np.asarray(fitted.scale_, dtype=np.float64)
            scales[~np.isfinite(scales) | (scales == 0)] = 1.0
            if self.mode == "numeric":
                # DictVectorizer generates name=value exclusively for string categoricals.
                categorical = np.fromiter(("=" in n for n in feature_names), bool, count=len(feature_names))
                scales[categorical] = 1.0
        factors = 1.0 / scales
        if self.category_weight != 1.0:
            categorical = np.fromiter(("=" in n for n in feature_names), bool, count=len(feature_names))
            factors[categorical] *= self.category_weight
        self.factors_ = factors.astype(np.float32)
        return self

    def transform(self, X: sparse.spmatrix) -> sparse.csr_matrix:
        if self.factors_ is None:
            raise RuntimeError("Scaler not fitted")
        out = X.tocsr().astype(np.float32, copy=True)
        inplace_csr_column_scale(out, self.factors_)
        return out


class NeighborEngine:
    """Find max_k neighbors once, reused for every candidate k.

    The optional CUDA path follows the notebook's exact dense CuPy brute-force
    distance formula. In 'auto' mode it safely falls back to the CPU if the
    CUDA installation or memory is insufficient.
    """

    def __init__(self, backend: str = "auto", gpu_batch_size: int = 64, n_jobs: int = -1):
        if backend not in ("auto", "cpu", "cuda"):
            raise ValueError("backend must be auto, cpu, or cuda")
        self.backend = backend
        self.gpu_batch_size = gpu_batch_size
        self.n_jobs = n_jobs
        self._cp: Any | None = None
        self.last_backend = "cpu"

    def _get_cupy(self):
        if self.backend == "cpu":
            return None
        if self._cp is not None:
            return self._cp
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("No NVIDIA CUDA device detected")
            self._cp = cp
            return cp
        except Exception as exc:
            if self.backend == "cuda":
                raise RuntimeError("CUDA selected but CuPy/CUDA is unavailable") from exc
            LOG.info("CuPy/CUDA unavailable: using CPU (%s)", exc)
            return None

    def _cpu(self, X_train, X_query, max_k: int) -> np.ndarray:
        nn = NearestNeighbors(
            n_neighbors=max_k,
            algorithm="brute",
            metric="euclidean",
            n_jobs=self.n_jobs,
        )
        nn.fit(X_train)
        return nn.kneighbors(X_query, return_distance=False).astype(np.int32)

    def _cuda(self, cp, X_train, X_query, max_k: int) -> np.ndarray:
        # Dense conversion is an important limitation of the original notebook.
        # Refuse to allocate when the estimated footprint exceeds available VRAM.
        n_train, n_feat = X_train.shape
        batch = min(self.gpu_batch_size, X_query.shape[0])
        bytes_expected = 4 * (n_train * n_feat + batch * n_feat + batch * n_train)
        free_bytes = cp.cuda.runtime.memGetInfo()[0]
        if bytes_expected * 2.0 + 128 * 1024**2 > free_bytes:
            raise MemoryError(
                f"CUDA preflight: dense matrices need at least ~{bytes_expected/1024**2:.0f} MiB "
                f"plus workspace; free VRAM {free_bytes/1024**2:.0f} MiB"
            )
        result = np.empty((X_query.shape[0], max_k), dtype=np.int32)
        try:
            train_gpu = cp.asarray(X_train.toarray(), dtype=cp.float32)
            train_sq = cp.sum(train_gpu * train_gpu, axis=1)
            for start in range(0, X_query.shape[0], self.gpu_batch_size):
                stop = min(start + self.gpu_batch_size, X_query.shape[0])
                query_gpu = cp.asarray(X_query[start:stop].toarray(), dtype=cp.float32)
                d2 = cp.sum(query_gpu * query_gpu, axis=1)[:, None] + train_sq[None, :]
                d2 -= 2.0 * (query_gpu @ train_gpu.T)
                cp.maximum(d2, 0.0, out=d2)
                candidate = cp.argpartition(d2, kth=max_k-1, axis=1)[:, :max_k]
                d2_subset = cp.take_along_axis(d2, candidate, axis=1)
                order = cp.argsort(d2_subset, axis=1)
                indices = cp.take_along_axis(candidate, order, axis=1)
                result[start:stop] = cp.asnumpy(indices).astype(np.int32)
                del query_gpu, d2, candidate, d2_subset, order, indices
            return result
        finally:
            # This is called even if allocation/compute failed.
            cp.get_default_memory_pool().free_all_blocks()

    def kneighbors(self, X_train, X_query, max_k: int) -> np.ndarray:
        if max_k <= 0 or max_k > X_train.shape[0]:
            raise ValueError(f"k={max_k} must be between 1 and {X_train.shape[0]}")
        cp = self._get_cupy()
        if cp is not None:
            try:
                result = self._cuda(cp, X_train, X_query, max_k)
                self.last_backend = "cuda"
                return result
            except Exception as exc:
                if self.backend == "cuda":
                    raise
                LOG.warning("GPU search failed (%s). Falling back to CPU.", exc)
                cp.get_default_memory_pool().free_all_blocks()
        self.last_backend = "cpu"
        return self._cpu(X_train, X_query, max_k)


class BattleKNNClassifier:
    """Serializable fitted classical kNN classifier."""

    def __init__(self, k: int, feature_set: str = "summary", scaling: str = "numeric",
                 category_weight: float = 1.0):
        self.k = k
        self.extractor = BattleFeatureExtractor(feature_set)
        self.scaling = scaling
        self.category_weight = category_weight
        self.vectorizer: DictVectorizer | None = None
        self.scaler: SparseFeatureScaler | None = None
        self.X_train: sparse.csr_matrix | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, records: list[dict], labels: np.ndarray):
        self.y_train = np.asarray(labels, dtype=np.int8)
        if len(records) != len(labels):
            raise ValueError("Records and labels length mismatch")
        if self.k > len(records):
            raise ValueError("k is greater than number of training examples")
        self.vectorizer = DictVectorizer(sparse=True, sort=True)
        X = self.vectorizer.fit_transform(self.extractor.transform(records)).tocsr().astype(np.float32)
        self.scaler = SparseFeatureScaler(self.scaling, self.category_weight)
        self.scaler.fit(X, self.vectorizer.get_feature_names_out().tolist())
        self.X_train = self.scaler.transform(X)
        return self

    def predict_proba(self, records: list[dict], engine: NeighborEngine) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.vectorizer is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        X_new = self.vectorizer.transform(self.extractor.transform(records)).tocsr().astype(np.float32)
        X_new = self.scaler.transform(X_new)
        neighbors = engine.kneighbors(self.X_train, X_new, self.k)
        return np.mean(self.y_train[neighbors], axis=1)

    def predict(self, records: list[dict], engine: NeighborEngine) -> np.ndarray:
        return (self.predict_proba(records, engine) > 0.5).astype(np.int8)
