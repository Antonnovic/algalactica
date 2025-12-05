"""
corrida_analisis.py

Módulo de análisis financiero para el proyecto Algaláctica.

Lee los costos y los ingresos detallados desde `costos_estimados.ods`
y construye proyecciones de resultados, flujos de caja y métricas
financieras (VAN/NPV, TIR/IRR, periodo de recuperación, etc.).

Hojas esperadas en `costos_estimados.ods`:

- 01_DESGLOSE_DE_LA_INVERSION
- 02_VARIABLES
- 03_COSTOS_PRODUCCION_PRODUCTO_A
- 04_COSTOS_DISTRIBUCION_PRODUCTO_A
- 05_COSTOS_ADMINISTRATIVOS_PRODUCTO_A
- 06_PRESUPUESTO_INGRESOS_PRODUCTO_A
- 07_PRESUPUESTO_INGRESOS_ADICIONALES
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades financieras
# ---------------------------------------------------------------------------


def npv(rate: float, cashflows: np.ndarray) -> float:
    """
    Valor actual neto (VAN / NPV).

    Parameters
    ----------
    rate : float
        Tasa de descuento en forma decimal (p.ej. 0.1354)
    cashflows : np.ndarray
        Array empezando en año 0
    """
    years = np.arange(len(cashflows), dtype=float)
    return float(np.sum(cashflows / (1.0 + rate) ** years))


def irr(cashflows: np.ndarray) -> Optional[float]:
    """
    TIR / IRR usando búsqueda numérica simple.
    Devuelve None si no se puede encontrar una raíz razonable.
    """
    # Necesitamos al menos un flujo negativo (inversión) y uno positivo
    if np.all(cashflows >= 0) or np.all(cashflows <= 0):
        return None

    # Búsqueda de la raíz del VAN(r) = 0 entre -0.9 y 5.0 (-90% a 500%)
    def f(rate: float) -> float:
        return npv(rate, cashflows)

    low, high = -0.9, 5.0
    f_low, f_high = f(low), f(high)
    if f_low * f_high > 0:
        # No hay cambio de signo; devolvemos None
        return None

    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return mid


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------


@dataclass
class DetailedCostModel:
    years: np.ndarray                  # [1,2,3,4,5]
    volume: np.ndarray                 # litros/año
    price_net: np.ndarray              # MXN/litro promedio
    ingresos_principales: np.ndarray   # MXN/año
    ingresos_adicionales: np.ndarray   # MXN/año
    total_ingresos: np.ndarray         # MXN/año

    var_cost_unit: float               # costo variable total MXN/litro
    fixed_cost_annual: float           # costo fijo total MXN/año
    discount_rate: float               # WACC / tasa mínimo atractiva
    capex_initial: float               # inversión inicial año 0 MXN
    depreciation_annual: float         # depreciación contable MXN/año


@dataclass
class Scenario:
    name: str
    years: np.ndarray
    volume: np.ndarray
    price_net: np.ndarray
    var_cost_unit: float
    fixed_cost_annual: float
    discount_rate: float
    capex_initial: float
    working_capital_initial: float = 0.0
    additional_capex_year3: float = 0.0
    salvage_value: float = 0.0
    depreciation_annual: float = 0.0


# ---------------------------------------------------------------------------
# Utilidades de lectura de la hoja ODS
# ---------------------------------------------------------------------------


def _read_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    """Lee una hoja del .ods usando el motor ODF."""
    return pd.read_excel(path, sheet_name=sheet_name, engine="odf")


def _find_row(df: pd.DataFrame, needle: str) -> int:
    """Devuelve el índice de la primera fila cuya col 0 contiene `needle`."""
    col0 = df.iloc[:, 0].astype(str).str.lower()
    idx = col0[col0.str.contains(needle.lower())].index
    if len(idx) == 0:
        raise ValueError(f"Texto '{needle}' no encontrado en la primera columna.")
    return idx[0]


def _parse_percentage(x) -> float:
    """Convierte algo como '12,77%' o 0.1277 en 0.1277 (float)."""
    if isinstance(x, str):
        x = x.replace("%", "").replace(",", ".")
        return float(x) / 100.0
    return float(x)


def _to_float(x: Any) -> float:
    """Intenta convertir cualquier cosa razonable en float."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        s = (
            x.replace("$", "")
            .replace(",", "")
            .replace("por litro", "")
            .replace("al día", "")
            .replace("al dia", "")
            .replace("anual", "")
            .strip()
        )
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Carga del modelo detallado desde costos_estimados.ods
# ---------------------------------------------------------------------------


def load_cost_model_from_excel(path: str = "costos_estimados.ods") -> DetailedCostModel:
    """
    Liest die Struktur von Kosten und Einnahmen aus `costos_estimados.ods`
    und baut daraus ein DetailedCostModel.

    Logik / Annahmen (angepasst an deine aktuelle .ods):

    - Inversión inicial & jährliche Abschreibung:
      Blatt `01_DESGLOSE_DE_LA_INVERSION`, Zeile "TOTAL".
    - Diskontsatz:
      Blatt `02_VARIABLES`, Zeile "Tasa interbancaria + 5% de premio".
    - Fixkosten:
      Summe aller Positionen mit Einheit „$/mes“, „S/mes“ etc. in
      Blättern 03–05, auf Jahresbasis (×12).
    - Variable Kosten pro Liter:
      aus der Zeile "TOTAL DE COSTOS" im Blatt
      `03_COSTOS_PRODUCCION_PRODUCTO_A` – dort die kleinere Zahl
      (≈ 48.28 MXN/litro).
    - Einnahmen:
      Hauptprodukt: Blatt `06_PRESUPUESTO_INGRESOS_PRODUCTO_A`.
      Zusätzliche: Blatt `07_PRESUPUESTO_INGRESOS_ADICIONALES`.
    """

    # ----------------- Inversión inicial & depreciación ---------------
    df_inv = _read_sheet(path, "01_DESGLOSE_DE_LA_INVERSION")
    idx_total_inv = _find_row(df_inv, "total")

    # Spalte mit "INVERSION" im Namen
    col_inv = next(
        (c for c in df_inv.columns if isinstance(c, str) and "INVERSION" in c.upper()),
        df_inv.columns[2],
    )
    capex_initial = _to_float(df_inv.loc[idx_total_inv, col_inv])

    # Spalte mit "Monto" o "Depreci" im Namen
    col_depr = next(
        (
            c
            for c in df_inv.columns
            if isinstance(c, str)
            and ("MONTO" in c.upper() or "DEPRECI" in c.upper())
        ),
        None,
    )
    if col_depr is not None:
        depreciation_annual = _to_float(df_inv.loc[idx_total_inv, col_depr])
    else:
        depreciation_annual = 0.0

    # ----------------- Variables & tasa de descuento -------------------
    df_var = _read_sheet(path, "02_VARIABLES")
    idx_tasa = _find_row(df_var, "interbancaria")  # "Tasa interbancaria + 5% de premio"

    col_tasa_base = next(
        (c for c in df_var.columns if isinstance(c, str) and "Tasa Base" in c),
        df_var.columns[2],
    )
    discount_rate = _parse_percentage(df_var.loc[idx_tasa, col_tasa_base])

    # ----------------- Fixkosten aus Blättern 03–05 -------------------
    sheets_costos = [
        "03_COSTOS_PRODUCCION_PRODUCTO_A",
        "04_COSTOS_DISTRIBUCION_PRODUCTO_A",
        "05_COSTOS_ADMINISTRATIVOS_PRODUCTO_A",
    ]

    total_fixed_cost_monthly: float = 0.0

    for sheet in sheets_costos:
        df = _read_sheet(path, sheet)

        col_unidad = next(
            (c for c in df.columns if isinstance(c, str) and "Unidad" in c),
            df.columns[1],
        )
        col_precio = next(
            (c for c in df.columns if isinstance(c, str) and "Precio" in c),
            df.columns[2],
        )

        # Relevante Zeilen (kein TOTAL, kein leerer CONCEPTO)
        mask_concepto = df.iloc[:, 0].notna() & ~df.iloc[:, 0].astype(
            str
        ).str.contains("TOTAL", case=False, na=False)
        df_costs = df.loc[mask_concepto].copy()

        unidad_str = df_costs[col_unidad].astype(str).str.lower()
        is_fixed = unidad_str.str.contains("mes")  # $/mes, S/mes, etc.

        fixed_monthly_sheet = (
            pd.to_numeric(df_costs.loc[is_fixed, col_precio], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        total_fixed_cost_monthly += float(fixed_monthly_sheet)

    fixed_cost_annual = 12.0 * total_fixed_cost_monthly

    # ----------------- Variable Kosten pro Liter (Blatt 03) -----------
    df_prod = _read_sheet(path, "03_COSTOS_PRODUCCION_PRODUCTO_A")
    try:
        idx_total_costs = _find_row(df_prod, "total de costos")
    except ValueError:
        try:
            idx_total_costs = _find_row(df_prod, "costos de producción")
        except ValueError:
            idx_total_costs = None

    var_cost_unit = 0.0
    if idx_total_costs is not None:
        row_tot = df_prod.loc[idx_total_costs]
        numeric_vals = pd.to_numeric(row_tot, errors="coerce").dropna()
        if not numeric_vals.empty:
            # Annahme: "por litro" ist im Hunderterbereich,
            # Tages- / Jahreswerte sind >> 1.000
            small_vals = numeric_vals[numeric_vals < 1_000]
            if not small_vals.empty:
                var_cost_unit = float(small_vals.max())
            else:
                var_cost_unit = float(numeric_vals.min())

    # ----------------- Ingresos principales ---------------------------
    df_ing = _read_sheet(path, "06_PRESUPUESTO_INGRESOS_PRODUCTO_A")

    idx_vol = _find_row(df_ing, "capacidad anual")
    idx_tot_ing = _find_row(df_ing, "total ingresos del producto a")

    year_cols = df_ing.columns[2:7]  # 5 años

    volume = df_ing.loc[idx_vol, year_cols].astype(float).to_numpy()
    ingresos_principales = df_ing.loc[idx_tot_ing, year_cols].astype(float).to_numpy()

    # ----------------- Ingresos adicionales ---------------------------
    try:
        df_ing_add = _read_sheet(path, "07_PRESUPUESTO_INGRESOS_ADICIONALES")
        idx_tot_add = _find_row(df_ing_add, "total ingresos del producto a")
        year_cols_add = df_ing_add.columns[2 : 2 + len(year_cols)]
        ingresos_adicionales = (
            df_ing_add.loc[idx_tot_add, year_cols_add].astype(float).to_numpy()
        )
    except Exception:
        ingresos_adicionales = np.zeros_like(ingresos_principales)

    total_ingresos = ingresos_principales + ingresos_adicionales

    # ----------------- Precio neto por litro --------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        price_net = np.where(volume > 0, total_ingresos / volume, 0.0)

    years = np.arange(1, len(volume) + 1, dtype=int)

    # Falls Depreciación aus irgendwelchen Gründen NaN wurde → 0
    if isinstance(depreciation_annual, float) and np.isnan(depreciation_annual):
        depreciation_annual = 0.0

    return DetailedCostModel(
        years=years,
        volume=volume,
        price_net=price_net,
        ingresos_principales=ingresos_principales,
        ingresos_adicionales=ingresos_adicionales,
        total_ingresos=total_ingresos,
        var_cost_unit=float(var_cost_unit),
        fixed_cost_annual=float(fixed_cost_annual),
        discount_rate=float(discount_rate),
        capex_initial=float(capex_initial),
        depreciation_annual=float(depreciation_annual),
    )


# ---------------------------------------------------------------------------
# Proyecciones y flujos de caja
# ---------------------------------------------------------------------------


def compute_projection(
    years: np.ndarray,
    volume: np.ndarray,
    price_net: np.ndarray,
    var_cost_unit: float,
    fixed_cost_annual: float,
    discount_rate: float,
    capex_initial: float,
    working_capital_initial: float = 0.0,
    additional_capex_by_year: Optional[Dict[int, float]] = None,
    depreciation_annual: float = 0.0,
    salvage_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Construye una proyección simple de 5 años para un solo producto.

    Devuelve un diccionario con tablas de resultados y métricas.
    """
    additional_capex_by_year = additional_capex_by_year or {}

    years = np.asarray(years, dtype=int)
    n = len(years)
    volume = np.asarray(volume, dtype=float)
    price_net = np.asarray(price_net, dtype=float)

    ingresos = volume * price_net
    costo_variable = volume * var_cost_unit
    costo_fijo = np.full(n, fixed_cost_annual, dtype=float)

    margen_bruto = ingresos - costo_variable
    utilidad_operativa = margen_bruto - costo_fijo
    utilidad_neta = utilidad_operativa - depreciation_annual  # sin impuestos detallados

    # ----------------- Flujos de caja del proyecto ----------------------
    cashflows = np.zeros(n + 1, dtype=float)
    cashflows[0] = -capex_initial - working_capital_initial

    for i, year in enumerate(years, start=1):
        capex_extra = additional_capex_by_year.get(int(year), 0.0)
        # Flujo = U.neta + Depreciación (no monetaria) - CAPEX extra
        cashflows[i] = utilidad_neta[i - 1] + depreciation_annual - capex_extra

    # Recuperación del capital de trabajo + valor de rescate en el último año
    cashflows[-1] += working_capital_initial + salvage_value

    van = npv(discount_rate, cashflows)
    tir = irr(cashflows)

    # Payback simple (año donde el flujo acumulado pasa de negativo a positivo)
    cf_cum = np.cumsum(cashflows)
    payback_year = None
    for i in range(1, len(cf_cum)):
        if cf_cum[i] >= 0:
            payback_year = years[i - 1]  # año del cambio de signo
            break

    return {
        "years": years,
        "volume": volume,
        "ingresos": ingresos,
        "costo_variable": costo_variable,
        "costo_fijo": costo_fijo,
        "margen_bruto": margen_bruto,
        "utilidad_operativa": utilidad_operativa,
        "utilidad_neta": utilidad_neta,
        "cashflows": cashflows,
        "cashflows_cumul": cf_cum,
        "npv": van,
        "irr": tir,
        "payback_year": payback_year,
    }


def build_cashflows_for_project(esc: Scenario) -> Dict[str, np.ndarray]:
    """
    Wrapper conveniente que toma un `Scenario` y llama a `compute_projection`.
    """
    add_capex: Dict[int, float] = {}
    if esc.additional_capex_year3:
        add_capex[3] = esc.additional_capex_year3

    return compute_projection(
        years=esc.years,
        volume=esc.volume,
        price_net=esc.price_net,
        var_cost_unit=esc.var_cost_unit,
        fixed_cost_annual=esc.fixed_cost_annual,
        discount_rate=esc.discount_rate,
        capex_initial=esc.capex_initial,
        working_capital_initial=esc.working_capital_initial,
        additional_capex_by_year=add_capex,
        depreciation_annual=esc.depreciation_annual,
        salvage_value=esc.salvage_value,
    )


# ---------------------------------------------------------------------------
# CLI clásico (para usar con `%run corrida_analisis.py --scenario base ...`)
# ---------------------------------------------------------------------------


def _create_base_scenario_from_excel(path: str = "costos_estimados.ods") -> Scenario:
    """
    Crea un escenario "base" utilizando directamente el modelo detallado
    leído desde la hoja de cálculo.
    """
    model = load_cost_model_from_excel(path)
    return Scenario(
        name="base",
        years=model.years,
        volume=model.volume,
        price_net=model.price_net,
        var_cost_unit=model.var_cost_unit,
        fixed_cost_annual=model.fixed_cost_annual,
        discount_rate=model.discount_rate,
        capex_initial=model.capex_initial,
        depreciation_annual=model.depreciation_annual,
    )


def run_scenario(name: str, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """
    Ejecuta un escenario por CLI y devuelve los resultados.
    """
    if name != "base":
        raise ValueError("Por ahora solo se implementa el escenario 'base'.")

    esc = _create_base_scenario_from_excel(args.costos_ods)

    res = build_cashflows_for_project(esc)

    # Impresión de tabla tipo "corrida" para el terminal
    print("==== Escenario:", esc.name.upper(), "====")
    print("Año  Volumen   Ingresos   CostVar   CostFijo   U.Op   U.Neta   Flujo")
    for i, year in enumerate(esc.years, start=1):
        print(
            f"{year:>3d} "
            f"{res['volume'][i-1]:>10,.0f} "
            f"{res['ingresos'][i-1]:>10,.0f} "
            f"{res['costo_variable'][i-1]:>10,.0f} "
            f"{res['costo_fijo'][i-1]:>10,.0f} "
            f"{res['utilidad_operativa'][i-1]:>10,.0f} "
            f"{res['utilidad_neta'][i-1]:>10,.0f} "
            f"{res['cashflows'][i]:>10,.0f}"
        )

    print("\nCashflows del proyecto (incl. CAPEX y capital de trabajo en año 0):")
    for i, cf in enumerate(res["cashflows"]):
        print(f"Año {i:>2d}: {cf:>12,.0f} MXN")

    irr_pct = res["irr"] * 100 if res["irr"] is not None else float("nan")
    print(f"\nTIR (IRR) estimada del proyecto: {irr_pct:.2f} %")
    print(f"VAN (NPV) al {esc.discount_rate*100:.2f} %: {res['npv']:,.0f} MXN")
    if res["payback_year"] is not None:
        print(f"Periodo de recuperación (payback): año {res['payback_year']}")
    else:
        print("Periodo de recuperación (payback): no se recupera en el horizonte.")

    return res


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Corrida financiera del proyecto Algaláctica"
    )
    parser.add_argument(
        "--scenario",
        default="base",
        help="Nombre del escenario (por ahora solo 'base')",
    )
    parser.add_argument(
        "--costos-ods",
        default="costos_estimados.ods",
        help="Ruta a la hoja de cálculo de costos (ODS)",
    )

    args = parser.parse_args(argv)
    run_scenario(args.scenario, args)


if __name__ == "__main__":  # pragma: no cover
    main()
