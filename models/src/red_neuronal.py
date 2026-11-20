"""
red_neuronal.py
MLP para clasificación binaria tabular (PyTorch), con early stopping,
historial de entrenamiento y reentrenamiento final en train+val.

Uso:
    from red_neuronal import ClasificadorMLP
    clf = ClasificadorMLP(capas=(128, 64, 32), dropout=0.3)
    clf.fit(X_tr, y_tr, X_va, y_va)
    clf.graficar_historial()
    print(clf.evaluar(X_va, y_va))
    pred = clf.predecir(X_te)            # booleans
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def fijar_semilla(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _a_tensor(X) -> torch.Tensor:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy(dtype=np.float32)
    return torch.tensor(np.array(X, dtype=np.float32))  # copia: evita warning de arrays read-only


class MLP(nn.Module):
    """Bloques Linear -> BatchNorm -> ReLU -> Dropout, salida = 1 logit."""

    def __init__(self, n_entradas: int, capas=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        bloques, n_prev = [], n_entradas
        for n in capas:
            bloques += [nn.Linear(n_prev, n), nn.BatchNorm1d(n), nn.ReLU(), nn.Dropout(dropout)]
            n_prev = n
        bloques.append(nn.Linear(n_prev, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x).squeeze(-1)


class ClasificadorMLP:
    def __init__(
        self,
        capas=(128, 64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epocas: int = 200,
        paciencia: int = 20,
        umbral: float = 0.5,
        seed: int = 42,
        device: str | None = None,
        verbose: int = 10,
    ):
        self.capas, self.dropout = capas, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epocas, self.paciencia = batch_size, max_epocas, paciencia
        self.umbral, self.seed, self.verbose = umbral, seed, verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo: MLP | None = None
        self.columnas: list[str] | None = None
        self.historial: dict[str, list[float]] = {}
        self.mejor_epoca: int | None = None

    # ------------------------------------------------------------------ #
    def _nuevo_modelo(self, n_entradas: int):
        fijar_semilla(self.seed)
        self.modelo = MLP(n_entradas, self.capas, self.dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.modelo.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=max(self.paciencia // 3, 3))
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _loader(self, X, y, barajar: bool):
        g = torch.Generator().manual_seed(self.seed)
        ds = TensorDataset(_a_tensor(X), _a_tensor(y))
        # drop_last evita un batch de 1 elemento, que rompe BatchNorm
        return DataLoader(ds, batch_size=self.batch_size, shuffle=barajar,
                          generator=g, drop_last=barajar and len(ds) % self.batch_size == 1)

    def _epoca_entrenamiento(self, loader) -> tuple[float, float]:
        self.modelo.train()
        perdida, aciertos, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.opt.zero_grad()
            logits = self.modelo(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.opt.step()
            perdida += loss.item() * len(xb)
            aciertos += ((torch.sigmoid(logits) > self.umbral) == yb.bool()).sum().item()
            n += len(xb)
        return perdida / n, aciertos / n

    @torch.no_grad()
    def _evaluar_loader(self, loader) -> tuple[float, float]:
        self.modelo.eval()
        perdida, aciertos, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.modelo(xb)
            perdida += self.loss_fn(logits, yb).item() * len(xb)
            aciertos += ((torch.sigmoid(logits) > self.umbral) == yb.bool()).sum().item()
            n += len(xb)
        return perdida / n, aciertos / n

    # ------------------------------------------------------------------ #
    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        """Entrena con early stopping sobre val_loss y restaura los mejores pesos."""
        if isinstance(X_tr, pd.DataFrame):
            self.columnas = list(X_tr.columns)
        self._nuevo_modelo(X_tr.shape[1])
        tr_loader = self._loader(X_tr, y_tr, barajar=True)
        va_loader = self._loader(X_va, y_va, barajar=False) if X_va is not None else None

        self.historial = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "lr": []}
        mejor_loss, mejor_pesos, sin_mejora = float("inf"), None, 0

        for ep in range(1, self.max_epocas + 1):
            loss, acc = self._epoca_entrenamiento(tr_loader)
            self.historial["loss"].append(loss)
            self.historial["acc"].append(acc)
            self.historial["lr"].append(self.opt.param_groups[0]["lr"])

            if va_loader is None:
                continue
            v_loss, v_acc = self._evaluar_loader(va_loader)
            self.historial["val_loss"].append(v_loss)
            self.historial["val_acc"].append(v_acc)
            self.sched.step(v_loss)

            if v_loss < mejor_loss - 1e-4:
                mejor_loss, sin_mejora = v_loss, 0
                mejor_pesos = copy.deepcopy(self.modelo.state_dict())
                self.mejor_epoca = ep
            else:
                sin_mejora += 1

            if self.verbose and (ep % self.verbose == 0 or ep == 1):
                print(f"época {ep:3d} | loss {loss:.4f} acc {acc:.4f} | "
                      f"val_loss {v_loss:.4f} val_acc {v_acc:.4f} | lr {self.historial['lr'][-1]:.1e}")
            if sin_mejora >= self.paciencia:
                if self.verbose:
                    print(f"Early stopping en época {ep}. Mejor época: {self.mejor_epoca} "
                          f"(val_loss {mejor_loss:.4f})")
                break

        if mejor_pesos is not None:
            self.modelo.load_state_dict(mejor_pesos)
        return self

    def reentrenar_completo(self, X_tr, y_tr, X_va, y_va):
        """Reentrena desde cero con train+val durante el número de épocas que fue óptimo."""
        if self.mejor_epoca is None:
            raise RuntimeError("Primero corre fit() con validación para conocer la mejor época.")
        X = pd.concat([X_tr, X_va]) if isinstance(X_tr, pd.DataFrame) else np.vstack([X_tr, X_va])
        y = pd.concat([y_tr, y_va]) if isinstance(y_tr, pd.Series) else np.concatenate([y_tr, y_va])
        epocas = self.mejor_epoca
        self._nuevo_modelo(X.shape[1])
        loader = self._loader(X, y, barajar=True)
        for _ in range(epocas):
            self._epoca_entrenamiento(loader)
        if self.verbose:
            print(f"Reentrenado con {len(X)} registros durante {epocas} épocas.")
        return self

    # ------------------------------------------------------------------ #
    def _validar_columnas(self, X):
        if self.columnas is not None and isinstance(X, pd.DataFrame):
            if list(X.columns) != self.columnas:
                faltan = set(self.columnas) - set(X.columns)
                sobran = set(X.columns) - set(self.columnas)
                raise ValueError(f"Columnas distintas a las de entrenamiento. "
                                 f"Faltan: {faltan or '-'} | Sobran: {sobran or '-'}")

    @torch.no_grad()
    def predecir_proba(self, X) -> np.ndarray:
        self._validar_columnas(X)
        self.modelo.eval()
        logits = self.modelo(_a_tensor(X).to(self.device))
        return torch.sigmoid(logits).cpu().numpy()

    def predecir(self, X) -> np.ndarray:
        return self.predecir_proba(X) > self.umbral

    def evaluar(self, X, y) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        y = np.asarray(y).astype(int)
        p = self.predecir_proba(X)
        return {"accuracy": accuracy_score(y, p > self.umbral),
                "f1": f1_score(y, p > self.umbral),
                "roc_auc": roc_auc_score(y, p)}

    # ------------------------------------------------------------------ #
    def graficar_historial(self):
        import matplotlib.pyplot as plt
        h = self.historial
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h["loss"], label="train")
        ax[1].plot(h["acc"], label="train")
        if h.get("val_loss"):
            ax[0].plot(h["val_loss"], label="val")
            ax[1].plot(h["val_acc"], label="val")
            for a in ax:
                a.axvline(self.mejor_epoca - 1, color="gray", ls="--", lw=0.8, label="mejor época")
        ax[0].set_title("Loss (BCE)")
        ax[1].set_title("Accuracy")
        for a in ax:
            a.set_xlabel("época")
            a.legend()
        plt.tight_layout()
        plt.show()

    def guardar(self, ruta: str = "modelo_mlp.pt"):
        torch.save({"estado": self.modelo.state_dict(), "columnas": self.columnas,
                    "config": {"capas": self.capas, "dropout": self.dropout,
                               "umbral": self.umbral},
                    "n_entradas": self.modelo.red[0].in_features}, ruta)

    @classmethod
    def cargar(cls, ruta: str = "modelo_mlp.pt", device: str | None = None):
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        clf = cls(**ck["config"], device=device)
        clf.modelo = MLP(ck["n_entradas"], clf.capas, clf.dropout).to(clf.device)
        clf.modelo.load_state_dict(ck["estado"])
        clf.columnas = ck["columnas"]
        return clf