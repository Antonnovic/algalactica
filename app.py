import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import corrida_analisis as ca
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evaluación económica de Algaláctica",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Lectura y edición de hojas ODS (Tablas 1–7)
# ---------------------------------------------------------------------------

ODS_PATH = "costos_estimados.ods"


@st.cache_data
def leer_hojas_ods(path: str) -> Dict[str, pd.DataFrame]:
    """Carga las hojas clave de costos_estimados.ods como DataFrames."""
    sheet_names = [
        "01_DESGLOSE_DE_LA_INVERSION",
        "02_VARIABLES",
        "03_COSTOS_PRODUCCION_PRODUCTO_A",
        "04_COSTOS_DISTRIBUCION_PRODUCTO_A",
        "05_COSTOS_ADMINISTRATIVOS_PRODUCTO_A",
        "06_PRESUPUESTO_INGRESOS_PRODUCTO_A",
        "07_PRESUPUESTO_INGRESOS_ADICIONALES",
    ]
    xls = pd.ExcelFile(path, engine="odf")
    hojas: Dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        df = xls.parse(name)
        # Añadimos una columna de comentarios si no existe
        if "Comentario" not in df.columns:
            df["Comentario"] = ""
        hojas[name] = df
    return hojas


def construir_modelo_costos_desde_hojas(
    hojas: Dict[str, pd.DataFrame]
) -> ca.DetailedCostModel:
    """
    Reconstruye DetailedCostModel usando las hojas que están en memoria.

    Monkey-patch de ca._read_sheet para que load_cost_model_from_excel
    use estas hojas en lugar de leer de disco.
    """
    original_read_sheet = ca._read_sheet

    def fake_read_sheet(path: str, sheet_name: str) -> pd.DataFrame:
        if sheet_name in hojas:
            return hojas[sheet_name].copy()
        return original_read_sheet(path, sheet_name)

    ca._read_sheet = fake_read_sheet  # type: ignore[attr-defined]
    try:
        model = ca.load_cost_model_from_excel("ignored.ods")
    finally:
        ca._read_sheet = original_read_sheet  # type: ignore[attr-defined]
    return model


# Inicializamos las hojas en sesión
if "hojas_ods" not in st.session_state:
    st.session_state["hojas_ods"] = leer_hojas_ods(ODS_PATH)

hojas_ods: Dict[str, pd.DataFrame] = st.session_state["hojas_ods"]

# Construimos el modelo de costos con las hojas actuales
modelo_costos: ca.DetailedCostModel = construir_modelo_costos_desde_hojas(hojas_ods)

years = modelo_costos.years
volumen_base = modelo_costos.volume
precio_promedio_base = modelo_costos.price_net
var_cost_unit_base = modelo_costos.var_cost_unit
fixed_cost_annual_base = modelo_costos.fixed_cost_annual
discount_rate_base = modelo_costos.discount_rate
capex_initial_default = modelo_costos.capex_initial
depreciacion_default = modelo_costos.depreciation_annual


# ---------------------------------------------------------------------------
# Layout principal: título + logo
# ---------------------------------------------------------------------------
col_title, col_logo = st.columns([3, 1])

with col_title:
    st.title("Evaluación económica de Algaláctica")

    st.markdown(
        """
Esta herramienta interactiva usa tu modelo financiero en Python
(**`corrida_analisis.py`**) y los supuestos detallados de costos e ingresos
guardados en **`costos_estimados.ods`**.

- En la barra lateral defines el **escenario agregado** (volúmenes, precios,
  costos totales, CAPEX, tasa de descuento, etc.).
- En la pestaña **“Supuestos detallados (Tablas 1–7)”** puedes revisar y
  anotar comentarios sobre cada hoja de cálculo.
- En la pestaña **“Resultados (Tablas 8–11)”** se muestran estado de resultados,
  flujos de caja, punto de equilibrio y métricas financieras.
- Opcional: ejecuta una **simulación de Monte Carlo** para ver la dispersión de
  la TIR y el VAN.
"""
    )

with col_logo:
    try:
        st.image(
            "figuras/Logotipo_Algalactica.png",
            use_container_width=True,
        )
    except Exception:
        st.write(" ")


st.markdown("---")


# ---------------------------------------------------------------------------
# Controles agregados de escenario (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Supuestos del escenario")

st.sidebar.subheader("Volúmenes anuales de ventas [L/año]")
vol_inputs = []
for i, y in enumerate(years):
    vol = float(
        st.sidebar.number_input(
            f"Volumen año {int(y)}",
            min_value=0.0,
            value=float(volumen_base[i]),
            step=1000.0,
        )
    )
    vol_inputs.append(vol)
vol_inputs = np.array(vol_inputs, dtype=float)

st.sidebar.subheader("Precios promedio [MXN/L]")
price_inputs = []
for i, y in enumerate(years):
    price = float(
        st.sidebar.number_input(
            f"Precio año {int(y)}",
            min_value=0.0,
            value=float(precio_promedio_base[i]),
            step=1.0,
            format="%.2f",
        )
    )
    price_inputs.append(price)
price_inputs = np.array(price_inputs, dtype=float)

st.sidebar.subheader("Costos variables y fijos")
var_cost_unit_input = float(
    st.sidebar.number_input(
        "Costo variable total por litro [MXN/L]",
        min_value=0.0,
        value=float(var_cost_unit_base),
        step=0.5,
        format="%.2f",
    )
)

fixed_cost_annual_input = float(
    st.sidebar.number_input(
        "Costo fijo total anual [MXN/año]",
        min_value=0.0,
        value=float(fixed_cost_annual_base),
        step=10000.0,
        format="%.2f",
    )
)

st.sidebar.subheader("Inversiones y tasa de descuento")
capex_initial = float(
    st.sidebar.number_input(
        "CAPEX inicial (año 0) [MXN]",
        min_value=0.0,
        value=float(capex_initial_default),
        step=50000.0,
        format="%.2f",
    )
)

working_capital_initial = float(
    st.sidebar.number_input(
        "Capital de trabajo inicial (año 0) [MXN]",
        min_value=0.0,
        value=50000.0,
        step=10000.0,
        format="%.2f",
    )
)

additional_capex_year3 = float(
    st.sidebar.number_input(
        "CAPEX adicional en año 3 [MXN]",
        min_value=0.0,
        value=200000.0,
        step=50000.0,
        format="%.2f",
    )
)

salvage_value = float(
    st.sidebar.number_input(
        "Valor de rescate al final del año 5 [MXN]",
        min_value=0.0,
        value=0.0,
        step=50000.0,
        format="%.2f",
    )
)

wacc_input = float(
    st.sidebar.number_input(
        "Tasa de descuento WACC efectiva [%]",
        min_value=0.0,
        max_value=40.0,
        value=float(discount_rate_base * 100.0),
        step=0.25,
        format="%.2f",
    )
)
discount_rate = wacc_input / 100.0


# ---------------------------------------------------------------------------
# Simulación de Monte Carlo – controles
# ---------------------------------------------------------------------------
st.sidebar.subheader("Simulación Monte Carlo")

enable_mc = st.sidebar.checkbox(
    "Activar simulación Monte Carlo (volumen y precio)",
    value=True,
)

n_sims = int(
    st.sidebar.number_input(
        "Número de corridas",
        min_value=100,
        max_value=20000,
        value=3000,
        step=100,
    )
)

sigma_vol_pct = float(
    st.sidebar.number_input(
        "Desviación volúmenes [%]",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        format="%.1f",
    )
)

sigma_price_pct = float(
    st.sidebar.number_input(
        "Desviación precios [%]",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.1f",
    )
)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def crear_escenario_interactivo() -> ca.Scenario:
    return ca.Scenario(
        name="Escenario interactivo",
        years=years,
        volume=vol_inputs,
        price_net=price_inputs,
        var_cost_unit=var_cost_unit_input,
        fixed_cost_annual=fixed_cost_annual_input,
        discount_rate=discount_rate,
        capex_initial=capex_initial,
        working_capital_initial=working_capital_initial,
        additional_capex_year3=additional_capex_year3,
        salvage_value=salvage_value,
        depreciation_annual=depreciacion_default,
    )


def calcular_break_even(scenario: ca.Scenario) -> pd.DataFrame:
    """
    Break-even simple por año:
    Q_BE = Costos fijos / (Precio - Costo variable unitario)
    """
    price = np.asarray(scenario.price_net, dtype=float)
    var_unit = float(scenario.var_cost_unit)
    fixed = float(scenario.fixed_cost_annual)

    margen_unit = price - var_unit
    q_be = np.where(margen_unit > 0, fixed / margen_unit, np.nan)

    return pd.DataFrame(
        {
            "Año": scenario.years,
            "Volumen BEQ [L/año]": q_be,
            "Margen unitario [MXN/L]": margen_unit,
        }
    )


def indicadores_financieros(
    res: Dict[str, Any], esc: ca.Scenario
) -> Dict[str, float]:
    """Calcula indicadores agregados a partir del resultado de build_cashflows_for_project."""
    cashflows = res["cashflows"]
    van = float(res["npv"])
    tir = res["irr"] if res["irr"] is not None else np.nan
    payback_year = res["payback_year"]

    # Índice de rentabilidad clásico: VAN / inversión inicial
    inv_inicial = abs(cashflows[0]) if cashflows[0] < 0 else 1.0
    indice_rent = van / inv_inicial

    # Valor terminal como perpetuidad del último flujo de caja
    if esc.discount_rate > 0:
        valor_terminal = cashflows[-1] / esc.discount_rate
        valor_terminal_pv = valor_terminal / (1.0 + esc.discount_rate) ** esc.years[-1]
    else:
        valor_terminal = np.nan
        valor_terminal_pv = np.nan

    return {
        "tir": tir,
        "van": van,
        "payback": payback_year if payback_year is not None else np.nan,
        "indice_rentabilidad": indice_rent,
        "valor_terminal": valor_terminal,
        "valor_terminal_pv": valor_terminal_pv,
    }


def ejecutar_monte_carlo(
    base_scenario: ca.Scenario,
    n_sims: int,
    sigma_vol_pct: float,
    sigma_price_pct: float,
    seed: int = 42,
):
    """
    Simulación Monte Carlo sobre volúmenes y precios.
    """
    rng = np.random.default_rng(seed)
    n_years = len(base_scenario.years)

    sigma_vol = sigma_vol_pct / 100.0
    sigma_price = sigma_price_pct / 100.0

    irr_list = []
    npv_list = []

    for _ in range(n_sims):
        vol_factor = 1.0 + rng.normal(0.0, sigma_vol, size=n_years)
        price_factor = 1.0 + rng.normal(0.0, sigma_price, size=n_years)

        vol_sim = np.maximum(0.0, base_scenario.volume * vol_factor)
        price_sim = np.maximum(0.0, base_scenario.price_net * price_factor)

        esc_sim = ca.Scenario(
            name="MC",
            years=base_scenario.years,
            volume=vol_sim,
            price_net=price_sim,
            var_cost_unit=base_scenario.var_cost_unit,
            fixed_cost_annual=base_scenario.fixed_cost_annual,
            discount_rate=base_scenario.discount_rate,
            capex_initial=base_scenario.capex_initial,
            working_capital_initial=base_scenario.working_capital_initial,
            additional_capex_year3=base_scenario.additional_capex_year3,
            salvage_value=base_scenario.salvage_value,
            depreciation_annual=base_scenario.depreciation_annual,
        )

        res_sim = ca.build_cashflows_for_project(esc_sim)
        irr_list.append(res_sim["irr"] if res_sim["irr"] is not None else np.nan)
        npv_list.append(res_sim["npv"])

    return np.array(irr_list, float), np.array(npv_list, float)


# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab_supuestos, tab_resultados, tab_mc = st.tabs(
    [
        "Supuestos detallados (Tablas 1–7)",
        "Resultados (Tablas 8–11)",
        "Monte Carlo",
    ]
)



# ---------------------------------------------------------------------------
# Pestaña de supuestos detallados – mostrar hojas 1–7
# ---------------------------------------------------------------------------
with tab_supuestos:
    st.header("Tablas 1–7 – Hojas de la ODS original")

    st.markdown(
        """
Aquí puedes revisar las hojas originales del archivo `costos_estimados.ods`.
Solo las celdas numéricas serán relevantes para los cálculos.  
Se ha añadido una columna **Comentario** para documentar ajustes o dudas.

> Nota: Streamlit no puede leer el color de las celdas (las celdas grises del
> Calc). Si quieres restringir la edición sólo a esas celdas, podemos afinar
> más adelante la lógica de columnas editables a mano.
"""
    )

    for nombre_hoja in hojas_ods.keys():
        df = hojas_ods[nombre_hoja]
        with st.expander(nombre_hoja, expanded=False):
            st.write(f"Vista previa / edición ligera de **{nombre_hoja}**")
            primera_col = df.columns[0]

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    primera_col: st.column_config.TextColumn(
                        primera_col,
                        disabled=True,
                    )
                },
                key=f"edit_{nombre_hoja}",
            )

            # Guardamos cambios en sesión para usarlos en el próximo cálculo
            hojas_ods[nombre_hoja] = edited_df

    st.info(
        "Cuando pulses **Calcular evaluación** en la pestaña de resultados, "
        "el modelo se recalculará usando estas hojas editadas."
    )


# ---------------------------------------------------------------------------
# Botón principal de cálculo (visible en pestaña de resultados)
# ---------------------------------------------------------------------------
with tab_resultados:
    st.header("Tablas 8–11 – Resultados del proyecto")

    calcular = st.button("Calcular evaluación")

    if calcular:
        escenario = crear_escenario_interactivo()
        resultado = ca.build_cashflows_for_project(escenario)
        st.session_state["ultimo_escenario"] = escenario
        st.session_state["ultimo_resultado"] = resultado

    escenario = st.session_state.get("ultimo_escenario")
    resultado = st.session_state.get("ultimo_resultado")

    if escenario is None or resultado is None:
        st.info("Configura los supuestos y pulsa **Calcular evaluación**.")
    else:
        # ----------------- Cuadro 5 / Tabla 8 – Estado de resultados ------
        st.subheader("Tabla 8 – Estado de resultados proyectado")

        df_proj = pd.DataFrame(
            {
                "Año": escenario.years,
                "Volumen [L/año]": resultado["volume"],
                "Ingresos [MXN/año]": resultado["ingresos"],
                "Costos variables [MXN/año]": resultado["costo_variable"],
                "Costos fijos [MXN/año]": resultado["costo_fijo"],
                "Utilidad bruta [MXN/año]": resultado["margen_bruto"],
                "Utilidad operativa [MXN/año]": resultado["utilidad_operativa"],
                "Utilidad neta [MXN/año]": resultado["utilidad_neta"],
            }
        )
        st.dataframe(df_proj.style.format(precision=2), use_container_width=True)

        # ----------------- Cuadro 6 / Tabla 9 – Flujos de caja -----------
        st.subheader("Tabla 9 – Flujos de caja del proyecto")

        cashflows = resultado["cashflows"]
        df_cf = pd.DataFrame(
            {
                "Periodo": np.arange(len(cashflows)),
                "Flujo de caja [MXN]": cashflows,
            }
        )
        st.dataframe(df_cf.style.format(precision=2), use_container_width=True)

        # ----------------- Cuadro 7 / Tabla 10 – Punto de equilibrio -----
        st.subheader("Tabla 10 – Punto de equilibrio (Break-even)")

        df_be = calcular_break_even(escenario)
        st.dataframe(df_be.style.format(precision=2), use_container_width=True)

        # ----------------- Indicadores financieros (similar a Cuadro 8) ---
        st.subheader("Tabla 11 – Indicadores financieros clave")

        ind = indicadores_financieros(resultado, escenario)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TIR (IRR)", f"{ind['tir'] * 100:,.2f} %")
            st.metric(
                "Índice de rentabilidad (PI)",
                f"{ind['indice_rentabilidad']:,.2f}",
            )
        with col2:
            st.metric("VAN (NPV) [MXN]", f"{ind['van']:,.0f}")
            st.metric(
                "Valor terminal (TV) [MXN]",
                f"{ind['valor_terminal']:,.0f}",
            )
        with col3:
            st.metric(
                "Periodo de recuperación (Payback) [años]",
                f"{ind['payback']:.2f}",
            )
            st.metric(
                "Valor presente del valor terminal [MXN]",
                f"{ind['valor_terminal_pv']:,.0f}",
            )

        # ----------------- Gráficos principales ---------------------------
        st.subheader("Gráficos principales")

        # 1) Ingresos vs Costos
        fig1, ax1 = plt.subplots()
        ax1.plot(
            df_proj["Año"],
            df_proj["Ingresos [MXN/año]"],
            label="Ingresos [MXN/año]",
        )
        ax1.plot(
            df_proj["Año"],
            df_proj["Costos variables [MXN/año]"],
            label="Costos variables [MXN/año]",
        )
        ax1.plot(
            df_proj["Año"],
            df_proj["Costos fijos [MXN/año]"],
            label="Costos fijos [MXN/año]",
        )
        ax1.set_xlabel("Año")
        ax1.set_ylabel("Monto [MXN]")
        ax1.set_title("Ingresos y costos")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # 2) Flujo de caja y flujo acumulado
        fig2, ax2 = plt.subplots()
        años_cf = np.arange(0, len(cashflows))
        ax2.bar(
            años_cf,
            cashflows,
            alpha=0.6,
            label="Flujo de caja [MXN]",
        )
        cf_acum = resultado["cashflows_cumul"]
        ax2.plot(
            años_cf,
            cf_acum,
            marker="o",
            label="Flujo de caja acumulado [MXN]",
        )
        ax2.set_xlabel("Periodo")
        ax2.set_ylabel("Monto [MXN]")
        ax2.set_title("Flujos de caja del proyecto")
        ax2.axhline(0, color="black", linewidth=1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)


# ---------------------------------------------------------------------------
# Pestaña Monte Carlo – usa el último escenario calculado
# ---------------------------------------------------------------------------
with tab_mc:
    st.header("Simulación Monte Carlo – TIR (IRR) y VAN (NPV)")

    escenario = st.session_state.get("ultimo_escenario")
    resultado = st.session_state.get("ultimo_resultado")

    if escenario is None or resultado is None:
        st.info(
            "Primero calcula un escenario base en la pestaña "
            "**Resultados (Tablas 8–11)**."
        )
    elif not enable_mc:
        st.info("Activa la opción de Monte Carlo en la barra lateral.")
    else:
        irr_sims, npv_sims = ejecutar_monte_carlo(
            escenario,
            n_sims=n_sims,
            sigma_vol_pct=sigma_vol_pct,
            sigma_price_pct=sigma_price_pct,
        )

        irr_sims_pct = irr_sims * 100.0

        col_mc1, col_mc2 = st.columns(2)

        with col_mc1:
            fig_mc1, ax_mc1 = plt.subplots()
            ax_mc1.hist(
                irr_sims_pct[~np.isnan(irr_sims_pct)],
                bins=40,
                alpha=0.8,
            )
            ax_mc1.set_xlabel("TIR simulada [%]")
            ax_mc1.set_ylabel("Frecuencia")
            ax_mc1.set_title("Distribución de TIR (IRR)")
            ax_mc1.grid(True, alpha=0.3)
            st.pyplot(fig_mc1)

            st.markdown(
                f"""
**TIR simulada – resumen**

- Media: {np.nanmean(irr_sims_pct):.2f} %
- Percentil 5: {np.nanpercentile(irr_sims_pct, 5):.2f} %
- Mediana (P50): {np.nanpercentile(irr_sims_pct, 50):.2f} %
- Percentil 95: {np.nanpercentile(irr_sims_pct, 95):.2f} %
"""
            )

        with col_mc2:
            fig_mc2, ax_mc2 = plt.subplots()
            ax_mc2.hist(
                npv_sims,
                bins=40,
                alpha=0.8,
            )
            ax_mc2.set_xlabel("VAN simulado [MXN]")
            ax_mc2.set_ylabel("Frecuencia")
            ax_mc2.set_title("Distribución de VAN (NPV)")
            ax_mc2.grid(True, alpha=0.3)
            st.pyplot(fig_mc2)

            st.markdown(
                f"""
**VAN simulado – resumen**

- Media: {np.mean(npv_sims):,.0f} MXN
- Percentil 5: {np.percentile(npv_sims, 5):,.0f} MXN
- Mediana (P50): {np.percentile(npv_sims, 50):,.0f} MXN
- Percentil 95: {np.percentile(npv_sims, 95):,.0f} MXN
"""
            )
