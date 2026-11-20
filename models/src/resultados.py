"""
resultados.py
Reporte de evaluación para clasificación binaria sobre el conjunto de validación.
Funciona con modelos de Keras (model.predict) y con ClasificadorMLP (predecir_proba).

Uso:
    from resultados import reporte_modelo
    res = reporte_modelo(model, X_va, y_va, history=history)
    res["umbral_optimo"]   # úsalo luego para predecir test
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def obtener_probabilidades(model, X) -> np.ndarray:
    """Devuelve P(clase=1) como vector 1D, sin importar el tipo de modelo."""
    X = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
    if hasattr(model, "predecir_proba"):            # ClasificadorMLP (PyTorch)
        return np.asarray(model.predecir_proba(X)).ravel()
    if hasattr(model, "predict_proba"):             # scikit-learn
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X, verbose=0)).ravel()   # Keras


def _metricas(y, proba, umbral) -> dict:
    pred = proba >= umbral
    return {
        "umbral": round(float(umbral), 3),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def _historial(history) -> dict | None:
    if history is None:
        return None
    if hasattr(history, "history"):                 # keras.callbacks.History
        return history.history
    if hasattr(history, "historial"):               # ClasificadorMLP
        return history.historial
    return dict(history)


def graficar_curvas(hist: dict, guardar: Path | None = None):
    metricas = [m for m in ("loss", "accuracy", "acc") if m in hist]
    fig, axes = plt.subplots(1, len(metricas), figsize=(6 * len(metricas), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metricas):
        ax.plot(hist[m], label="train")
        if f"val_{m}" in hist and len(hist[f"val_{m}"]):
            ax.plot(hist[f"val_{m}"], label="validación")
            mejor = int(np.argmin(hist["val_loss"])) if "val_loss" in hist else None
            if mejor is not None:
                ax.axvline(mejor, color="gray", ls="--", lw=0.8, label=f"mejor época ({mejor + 1})")
        ax.set_title(m)
        ax.set_xlabel("época")
        ax.legend()
    plt.suptitle("Curvas de aprendizaje")
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar / "curvas_aprendizaje.png", dpi=150)
    plt.show()


def reporte_modelo(
    model,
    X_val,
    y_val,
    history=None,
    umbral: float = 0.5,
    nombres_clases=("No transportado", "Transportado"),
    carpeta_salida: str | Path | None = None,
) -> dict:
    """
    Genera: curvas de aprendizaje, métricas, classification report, matriz de confusión
    (conteos y %), curva ROC con AUC, distribución de probabilidades y búsqueda del
    umbral que maximiza accuracy en validación.
    """
    guardar = Path(carpeta_salida) if carpeta_salida else None
    if guardar:
        guardar.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y_val).astype(int).ravel()
    proba = obtener_probabilidades(model, X_val)
    pred = proba >= umbral

    # 1) Curvas de aprendizaje
    hist = _historial(history)
    if hist:
        graficar_curvas(hist, guardar)

    # 2) Métricas y reporte por clase
    met = _metricas(y, proba, umbral)
    print("=" * 60)
    print(f"Validación: {len(y)} registros | positivos: {y.mean():.1%}")
    print("=" * 60)
    print(pd.Series(met).round(4).to_string(), "\n")
    print(classification_report(y, pred, target_names=list(nombres_clases), digits=4))

    # 3) Matriz de confusión + ROC + distribución de probabilidades
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=nombres_clases).plot(
        ax=ax[0], cmap="Blues", colorbar=False, values_format="d")
    ax[0].set_title("Matriz de confusión (conteos)")
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=nombres_clases).plot(
        ax=ax[1], cmap="Blues", colorbar=False, values_format=".1%")
    ax[1].set_title("Matriz de confusión (% por clase real)")

    fpr, tpr, umbrales_roc = roc_curve(y, proba)
    i_pt = int(np.argmin(np.abs(umbrales_roc - umbral)))
    ax[2].plot(fpr, tpr, lw=2, label=f"AUC = {met['roc_auc']:.4f}")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="azar")
    ax[2].scatter(fpr[i_pt], tpr[i_pt], c="red", zorder=3, label=f"umbral {umbral}")
    ax[2].set_xlabel("Tasa de falsos positivos")
    ax[2].set_ylabel("Tasa de verdaderos positivos (recall)")
    ax[2].set_title("Curva ROC")
    ax[2].legend(loc="lower right")

    ax[3].hist(proba[y == 0], bins=40, alpha=0.6, label=nombres_clases[0])
    ax[3].hist(proba[y == 1], bins=40, alpha=0.6, label=nombres_clases[1])
    ax[3].axvline(umbral, color="red", ls="--", lw=1)
    ax[3].set_xlabel("Probabilidad predicha")
    ax[3].set_title("Separación de clases")
    ax[3].legend()
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar / "evaluacion_validacion.png", dpi=150)
    plt.show()

    # 4) Umbral óptimo (accuracy) en validación
    candidatos = np.linspace(0.2, 0.8, 61)
    accs = [accuracy_score(y, proba >= u) for u in candidatos]
    u_opt = float(candidatos[int(np.argmax(accs))])
    met_opt = _metricas(y, proba, u_opt)
    print(f"Umbral óptimo en validación: {u_opt:.2f} "
          f"(accuracy {met_opt['accuracy']:.4f} vs {met['accuracy']:.4f} con {umbral})")
    if abs(met_opt["accuracy"] - met["accuracy"]) < 0.005:
        print("  -> diferencia mínima; quédate con 0.5 para no sobreajustar a validación.")

    tabla = pd.DataFrame([met, met_opt], index=[f"umbral {umbral}", "umbral óptimo"]).round(4)
    if guardar:
        tabla.to_csv(guardar / "metricas_validacion.csv")

    return {"metricas": met, "metricas_umbral_optimo": met_opt, "umbral_optimo": u_opt,
            "matriz_confusion": cm, "probabilidades": proba, "tabla": tabla}