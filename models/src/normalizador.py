"""
normalizador.py
Clase para cargar, inspeccionar, imputar, codificar y estandarizar datasets
(txt, JSON, csv, Excel) antes de entrenar redes neuronales.

Uso rápido:
    norm = NormalizadorDataset("datos.txt", delimitador="|")
    norm.preview()
    norm.correlacion(etapa="antes", heatmap=True)
    df_listo = norm.procesar(metodo_estandarizacion="zscore",
                             imputacion_categorica="moda")
    norm.correlacion(etapa="despues", heatmap=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class NormalizadorDataset:
    FORMATOS = {".txt", ".csv", ".json", ".xlsx", ".xls"}

    def __init__(
        self,
        ruta: str | Path,
        delimitador: str | None = None,
        umbral_dummies: int = 5,
        umbral_categorica_numerica: int = 10,
        columnas_excluir: list[str] | None = None,
        **kwargs_lectura,
    ):
        """
        ruta: archivo de entrada.
        delimitador: separador para .txt (y opcional para .csv). Ej: "\\t", "|", ";".
        umbral_dummies: si una categórica tiene MÁS de este número de categorías se dumifica;
                        si tiene este número o menos, se codifica con label encoding.
        umbral_categorica_numerica: una columna numérica entera con <= este número de
                        valores únicos se trata como categórica.
        columnas_excluir: columnas que no se tocan (p. ej. la variable objetivo).
        kwargs_lectura: se pasan directo a la función de pandas (sheet_name, encoding, etc.).
        """
        self.ruta = Path(ruta)
        self.delimitador = delimitador
        self.umbral_dummies = umbral_dummies
        self.umbral_cat_num = umbral_categorica_numerica
        self.columnas_excluir = set(columnas_excluir or [])

        self.df_original: pd.DataFrame = self._cargar(**kwargs_lectura)
        self.df: pd.DataFrame | None = None  # resultado procesado
        self.tipos: dict[str, str] = self._detectar_tipos(self.df_original)

        # Parámetros aprendidos (para aplicar después a datos nuevos)
        self.valores_imputacion: dict[str, object] = {}
        self.mapeos_label: dict[str, dict] = {}
        self.columnas_dummies: list[str] = []
        self.parametros_escala: dict[str, tuple[float, float]] = {}
        self.metodo_estandarizacion: str | None = None
        self.correlaciones: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # Carga
    # ------------------------------------------------------------------ #
    def _cargar(self, **kw) -> pd.DataFrame:
        ext = self.ruta.suffix.lower()
        if ext not in self.FORMATOS:
            raise ValueError(
                f"Formato no soportado: {ext}. Usa {sorted(self.FORMATOS)}"
            )

        if ext == ".txt":
            sep = self.delimitador if self.delimitador is not None else r"\s+"
            return pd.read_csv(self.ruta, sep=sep, engine="python", **kw)
        if ext == ".csv":
            return pd.read_csv(self.ruta, sep=self.delimitador or ",", **kw)
        if ext == ".json":
            try:
                return pd.read_json(self.ruta, **kw)
            except ValueError:  # JSON lines
                return pd.read_json(self.ruta, lines=True, **kw)
        return pd.read_excel(self.ruta, **kw)  # .xlsx / .xls

    # ------------------------------------------------------------------ #
    # Detección de tipos
    # ------------------------------------------------------------------ #
    def _detectar_tipos(self, df: pd.DataFrame) -> dict[str, str]:
        tipos = {}
        n = len(df)
        for col in df.columns:
            s = df[col]
            no_nulos = s.dropna()
            n_unicos = no_nulos.nunique()
            ratio_unicos = n_unicos / max(len(no_nulos), 1)
            nombre = str(col).lower()

            if len(no_nulos) == 0:
                tipos[col] = "vacía"
            elif pd.api.types.is_bool_dtype(s) or n_unicos == 2:
                tipos[col] = "binaria"
            elif pd.api.types.is_datetime64_any_dtype(s) or self._parece_fecha(
                no_nulos
            ):
                tipos[col] = "fecha"
            elif self._es_identificador(no_nulos, nombre, ratio_unicos, n):
                tipos[col] = "identificador"
            elif pd.api.types.is_numeric_dtype(s):
                es_entera = np.allclose(no_nulos, np.round(no_nulos))
                if es_entera and n_unicos <= self.umbral_cat_num:
                    tipos[col] = "categórica"
                elif es_entera:
                    tipos[col] = "discreta"
                else:
                    tipos[col] = "continua"
            else:
                # texto: categórica si se repite, texto libre si casi todo es único
                tipos[col] = (
                    "texto libre"
                    if ratio_unicos > 0.5 and n_unicos > 50
                    else "categórica"
                )
        return tipos

    @staticmethod
    def _parece_fecha(s: pd.Series) -> bool:
        if not pd.api.types.is_object_dtype(s):
            return False
        muestra = s.astype(str).head(50)
        parseadas = pd.to_datetime(muestra, errors="coerce", format="mixed")
        return parseadas.notna().mean() > 0.8

    @staticmethod
    def _es_identificador(s, nombre, ratio_unicos, n) -> bool:
        pistas = ("id", "uuid", "folio", "clave", "key", "codigo", "código")
        nombre_sugiere = any(
            nombre == p or nombre.endswith("_" + p) or nombre.startswith(p + "_")
            for p in pistas
        )
        if ratio_unicos < 0.95:
            return False
        if nombre_sugiere:
            return True
        if pd.api.types.is_integer_dtype(s):
            # enteros consecutivos o casi → índice
            v = np.sort(s.to_numpy())
            return len(v) > 1 and np.all(np.diff(v) == 1)
        return pd.api.types.is_object_dtype(s) and n > 20

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #
    def preview(self, n_filas: int = 5, mostrar: bool = True) -> pd.DataFrame:
        df = self.df_original
        resumen = pd.DataFrame(
            {
                "tipo_detectado": pd.Series(self.tipos),
                "dtype": df.dtypes.astype(str),
                "n_unicos": df.nunique(),
                "faltantes": df.isna().sum(),
                "%_faltantes": (df.isna().mean() * 100).round(2),
                "ejemplo": df.apply(
                    lambda c: c.dropna().iloc[0] if c.notna().any() else None
                ),
            }
        )
        resumen["acción"] = [self._accion_prevista(c) for c in resumen.index]
        if mostrar:
            print(
                f"Archivo: {self.ruta.name} | filas: {len(df)} | columnas: {df.shape[1]}\n"
            )
            print(df.head(n_filas).to_string(), "\n")
            print(resumen.to_string())
        return resumen

    def _accion_prevista(self, col) -> str:
        t = self.tipos[col]
        if col in self.columnas_excluir:
            return "excluida"
        if t in ("identificador", "texto libre", "vacía", "fecha"):
            return "se elimina"
        if t in ("categórica", "binaria"):
            n = self.df_original[col].nunique()
            return "dummies" if n > self.umbral_dummies else "label encoding"
        return "estandarizar"

    def reclasificar(self, **col_tipo: str) -> None:
        """Corrige a mano la detección. Ej: norm.reclasificar(edad="continua")"""
        for col, tipo in col_tipo.items():
            if col not in self.tipos:
                raise KeyError(col)
            self.tipos[col] = tipo

    # ------------------------------------------------------------------ #
    # Imputación
    # ------------------------------------------------------------------ #
    def _imputar(
        self, df: pd.DataFrame, imputacion_categorica: Literal["moda", "mediana"]
    ) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col in self.columnas_excluir or not df[col].isna().any():
                continue
            t = self.tipos.get(col)
            if t in ("categórica", "binaria"):
                valor = self._valor_categorico(df[col], imputacion_categorica)
            elif t in ("continua", "discreta"):
                valor = df[col].mean()
                if t == "discreta":
                    valor = round(valor)
            else:
                continue
            self.valores_imputacion[col] = valor
            df[col] = df[col].fillna(valor)
        return df

    @staticmethod
    def _valor_categorico(s: pd.Series, metodo: str):
        no_nulos = s.dropna()
        if metodo == "moda":
            return no_nulos.mode().iloc[0]
        if metodo == "mediana":
            # mediana sobre el orden de las categorías (útil si son ordinales)
            ordenadas = np.sort(no_nulos.to_numpy())
            return ordenadas[len(ordenadas) // 2]
        raise ValueError("imputacion_categorica debe ser 'moda' o 'mediana'")

    # ------------------------------------------------------------------ #
    # Codificación
    # ------------------------------------------------------------------ #
    def _codificar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        a_dumificar = []
        for col in list(df.columns):
            if col in self.columnas_excluir:
                continue
            if self.tipos.get(col) not in ("categórica", "binaria"):
                continue
            if df[col].nunique() > self.umbral_dummies:
                a_dumificar.append(col)
            else:
                categorias = sorted(df[col].dropna().unique(), key=str)
                mapeo = {c: i for i, c in enumerate(categorias)}
                self.mapeos_label[col] = mapeo
                df[col] = df[col].map(mapeo).astype(int)
        if a_dumificar:
            df = pd.get_dummies(df, columns=a_dumificar, prefix=a_dumificar, dtype=int)
            self.columnas_dummies = [
                c
                for c in df.columns
                if any(str(c).startswith(f"{p}_") for p in a_dumificar)
            ]
        return df

    # ------------------------------------------------------------------ #
    # Estandarización
    # ------------------------------------------------------------------ #
    def _estandarizar(
        self, df: pd.DataFrame, metodo: Literal["zscore", "media_varianza", "minmax"]
    ) -> pd.DataFrame:
        """
        zscore         -> (x - media) / desviación estándar
        media_varianza -> (x - media) / varianza
        minmax         -> (x - min) / (max - min)
        """
        df = df.copy()
        self.metodo_estandarizacion = metodo
        cols = [
            c
            for c in df.columns
            if self.tipos.get(c) in ("continua", "discreta")
            and c not in self.columnas_excluir
        ]
        for col in cols:
            x = df[col].astype(float)
            if metodo == "zscore":
                centro, escala = x.mean(), x.std(ddof=0)
            elif metodo == "media_varianza":
                centro, escala = x.mean(), x.var(ddof=0)
            elif metodo == "minmax":
                centro, escala = x.min(), x.max() - x.min()
            else:
                raise ValueError(
                    "metodo debe ser 'zscore', 'media_varianza' o 'minmax'"
                )
            escala = escala if escala and not np.isnan(escala) else 1.0
            self.parametros_escala[col] = (centro, escala)
            df[col] = (x - centro) / escala
        return df

    # ------------------------------------------------------------------ #
    # Pipeline completo
    # ------------------------------------------------------------------ #
    def procesar(
        self,
        metodo_estandarizacion: Literal[
            "zscore", "media_varianza", "minmax"
        ] = "zscore",
        imputacion_categorica: Literal["moda", "mediana"] = "moda",
        eliminar_no_utiles: bool = True,
    ) -> pd.DataFrame:
        df = self.df_original.copy()
        if eliminar_no_utiles:
            quitar = [
                c
                for c, t in self.tipos.items()
                if t in ("identificador", "texto libre", "vacía", "fecha")
                and c not in self.columnas_excluir
            ]
            df = df.drop(columns=quitar)
        df = self._imputar(df, imputacion_categorica)
        df = self._codificar(df)
        df = self._estandarizar(df, metodo_estandarizacion)
        self.df = df
        return df

    # ------------------------------------------------------------------ #
    # Correlación
    # ------------------------------------------------------------------ #
    def correlacion(
        self,
        etapa: Literal["antes", "despues"] = "antes",
        metodo: Literal["pearson", "spearman", "kendall"] = "pearson",
        heatmap: bool = True,
        anotar: bool | None = None,
        guardar: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if etapa == "antes":
            base = self.df_original.select_dtypes(include="number")
            base = base[
                [c for c in base.columns if self.tipos.get(c) != "identificador"]
            ]
        elif etapa == "despues":
            if self.df is None:
                raise RuntimeError(
                    "Llama a procesar() antes de pedir la correlación 'despues'."
                )
            base = self.df.select_dtypes(include="number")
        else:
            raise ValueError("etapa debe ser 'antes' o 'despues'")

        base = base.loc[:, base.nunique() > 1]  # columnas constantes dan NaN
        corr = base.corr(method=metodo)
        self.correlaciones[etapa] = corr

        if heatmap:
            n = len(corr)
            anotar = n <= 15 if anotar is None else anotar
            figsize = figsize or (max(6, n * 0.6), max(5, n * 0.5))
            plt.figure(figsize=figsize)
            mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(
                corr,
                mask=mascara,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                annot=anotar,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            plt.title(f"Correlación ({metodo}) — {etapa} de procesar")
            plt.tight_layout()
            if guardar:
                plt.savefig(guardar, dpi=150)
            plt.show()
        return corr

    def correlaciones_altas(
        self, etapa: str = "despues", umbral: float = 0.8
    ) -> pd.DataFrame:
        """Pares de variables con |r| >= umbral (útil para detectar redundancia)."""
        corr = self.correlaciones.get(etapa)
        if corr is None:
            corr = self.correlacion(etapa=etapa, heatmap=False)
        triu = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        pares = triu.stack().reset_index()
        pares.columns = ["var_1", "var_2", "r"]
        return pares[pares["r"].abs() >= umbral].sort_values(
            "r", key=abs, ascending=False
        )

    # ------------------------------------------------------------------ #
    # Aplicar a datos nuevos (test / producción) con los parámetros aprendidos
    # ------------------------------------------------------------------ #
    def transformar(self, df_nuevo: pd.DataFrame) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError(
                "Primero ejecuta procesar() sobre los datos de entrenamiento."
            )
        df = df_nuevo.copy()
        df = df[[c for c in df.columns if c in self.df_original.columns]]
        for col, valor in self.valores_imputacion.items():
            if col in df:
                df[col] = df[col].fillna(valor)
        for col, mapeo in self.mapeos_label.items():
            if col in df:
                df[col] = (
                    df[col].map(mapeo).fillna(-1).astype(int)
                )  # -1 = categoría nueva
        prefijos = {c.rsplit("_", 1)[0] for c in self.columnas_dummies}
        prefijos = [p for p in prefijos if p in df.columns]
        if prefijos:
            df = pd.get_dummies(df, columns=prefijos, prefix=prefijos, dtype=int)
        for col, (centro, escala) in self.parametros_escala.items():
            if col in df:
                df[col] = (df[col].astype(float) - centro) / escala
        return df.reindex(columns=self.df.columns, fill_value=0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python normalizador.py <archivo> [delimitador]")
        sys.exit(1)
    norm = NormalizadorDataset(
        sys.argv[1], delimitador=sys.argv[2] if len(sys.argv) > 2 else None
    )
    norm.preview()
    norm.correlacion("antes")
    norm.procesar()
    norm.correlacion("despues")
    print(norm.df.head())
