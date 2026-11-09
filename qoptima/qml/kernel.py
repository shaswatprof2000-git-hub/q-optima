"""Quantum-AI #1: a quantum-kernel classifier on real data.

Each data point x (a few features after PCA) is encoded into an n-qubit state with
the ZZ feature map of Havlicek et al., Nature 567, 209 (2019):

    |phi(x)> = ( U_Z(x) H^n )^reps |0...0>,
    U_Z(x)   = exp( i [ sum_i  b x_i Z_i  +  sum_(i,j) b^2 (pi - x_i)(pi - x_j) Z_i Z_j ] )

where b is a bandwidth factor. The kernel (similarity) of two points is the
overlap of their states, which a quantum computer estimates by sampling:

    k(x, x') = |<phi(x)|phi(x')>|^2

A classical support-vector machine then trains on that kernel. We compare it
with a classical RBF-kernel SVM and logistic regression on the same features.

Honest notes: on small, easy datasets like these, classical models usually do
as well or better. Quantum kernels also suffer "exponential concentration": with
many qubits all overlaps approach 0, so the kernel stops being informative.
The bandwidth b (Shaydulin & Wild, PRA 106, 042407, 2022) counters that. This is
a learning tool, not a medical diagnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def _hadamard_all(states: np.ndarray, n: int) -> np.ndarray:
    """Apply H to every qubit of a batch of states (shape: batch x 2^n)."""
    b = states.shape[0]
    for q in range(n):
        v = states.reshape(b, 2 ** (n - q - 1), 2, 2 ** q)
        a, c = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
        v[:, :, 0, :] = (a + c) / np.sqrt(2)
        v[:, :, 1, :] = (a - c) / np.sqrt(2)
    return states


@dataclass
class ZZFeatureMap:
    n_qubits: int
    reps: int = 2
    bandwidth: float = 1.0
    entanglement: str = "linear"   # "linear" (neighbours) or "full" (all pairs)

    def pairs(self):
        n = self.n_qubits
        if self.entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        return [(i, i + 1) for i in range(n - 1)]

    def states(self, X: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        X = np.asarray(X, dtype=float)
        if X.shape[1] != n:
            raise ValueError(f"expected {n} features, got {X.shape[1]}")
        k = np.arange(2 ** n)
        z = 1 - 2 * ((k[:, None] >> np.arange(n)) & 1)          # (2^n, n) of +/-1
        b = self.bandwidth
        phase = (b * X) @ z.T                                      # single-qubit terms
        for i, j in self.pairs():
            phase += (b ** 2) * ((np.pi - X[:, i]) * (np.pi - X[:, j]))[:, None] * (z[:, i] * z[:, j])[None, :]
        psi = np.zeros((X.shape[0], 2 ** n), dtype=np.complex128)
        psi[:, 0] = 1.0
        for _ in range(self.reps):
            psi = _hadamard_all(psi, n)
            psi = psi * np.exp(1j * phase)
        return psi

    def kernel(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        SA = self.states(A)
        SB = SA if B is None else self.states(B)
        return np.abs(SA.conj() @ SB.T) ** 2


def run_classification(dataset: str = "breast_cancer", n_qubits: int = 4, reps: int = 2,
                       bandwidths=(0.1, 0.25, 0.5, 1.0), test_size: float = 0.3,
                       seed: int = 0, max_train: int = 300) -> dict:
    """Train quantum-kernel and classical models on a real dataset; report test accuracy.

    Bandwidth is chosen by 5-fold cross-validation on the training set only, so the
    test set stays untouched."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.svm import SVC

    from ..datasets import classification_dataset

    X, y, _, desc = classification_dataset(dataset)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if len(Xtr) > max_train:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]

    std = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n_qubits, random_state=seed).fit(std.transform(Xtr))
    scale = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(std.transform(Xtr)))
    prep = lambda A: np.clip(scale.transform(pca.transform(std.transform(A))), 0, np.pi)  # noqa: E731
    Ftr, Fte = prep(Xtr), prep(Xte)

    # choose the bandwidth by cross-validation on the training data
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv_scores = {}
    for bw in bandwidths:
        fm = ZZFeatureMap(n_qubits, reps, bw)
        K = fm.kernel(Ftr)
        accs = []
        for a, b in cv.split(Ftr, ytr):
            m = SVC(kernel="precomputed", C=1.0).fit(K[np.ix_(a, a)], ytr[a])
            accs.append((m.predict(K[np.ix_(b, a)]) == ytr[b]).mean())
        cv_scores[bw] = float(np.mean(accs))
    best_bw = max(cv_scores, key=cv_scores.get)

    results = {}
    t0 = time.perf_counter()
    fm = ZZFeatureMap(n_qubits, reps, best_bw)
    Ktr, Kte = fm.kernel(Ftr), fm.kernel(Fte, Ftr)
    qsvm = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr)
    results["Quantum kernel SVM"] = {"accuracy": float((qsvm.predict(Kte) == yte).mean()),
                                     "seconds": time.perf_counter() - t0}
    for name, model in [("Classical RBF SVM", SVC(kernel="rbf", C=1.0, gamma="scale")),
                        ("Logistic regression", LogisticRegression(max_iter=1000))]:
        t0 = time.perf_counter()
        model.fit(Ftr, ytr)
        results[name] = {"accuracy": float(model.score(Fte, yte)), "seconds": time.perf_counter() - t0}
    results["Majority-class guess"] = {"accuracy": float(max(yte.mean(), 1 - yte.mean())), "seconds": 0.0}

    off = Ktr[~np.eye(len(Ktr), dtype=bool)]
    return {
        "dataset": desc, "n_qubits": n_qubits, "reps": reps, "train": len(Ftr), "test": len(Fte),
        "pca_variance_kept": float(pca.explained_variance_ratio_.sum()),
        "bandwidth_cv": cv_scores, "bandwidth": best_bw, "results": results,
        "kernel_offdiag_mean": float(off.mean()), "kernel_offdiag_std": float(off.std()),
        "kernel_train": Ktr, "y_train": ytr,
    }
