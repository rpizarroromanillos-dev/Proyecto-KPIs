"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PORTAL ÚNICO DE INTELIGENCIA DE NEGOCIO — v2.0                           ║
║   Arquitectura: RLS → Agente Enrutador → BI Estándar / Visual IA / Insights║
║   Stack: Python · Streamlit · Plotly Express · Claude API (claude-sonnet)  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  FLUJO DE DATOS:
    generar_datos() → aplicar_rls() → módulo seleccionado
                                       ├── BI Estándar   (gráficos fijos)
                                       ├── Visual IA     (Plotly generado por Claude)
                                       └── Insights IA   (narrativa C-Level por Claude)

  INTEGRACIÓN CLAUDE API:
    · Busca ANTHROPIC_API_KEY en variables de entorno
    · Si no existe, usa stubs deterministas para desarrollo local
    · Nunca se envían datos al modelo, solo cabeceras + estadísticas agregadas
"""

import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Claude SDK (opcional) ─────────────────────────────────────────────────────
try:
    import anthropic
    CLAUDE_DISPONIBLE = True
except ImportError:
    CLAUDE_DISPONIBLE = False

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURACIÓN GLOBAL & ESTILOS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="BI Portal · Multinacional",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta: azul pizarra corporativo + ámbar como único acento de alerta
C = {
    "bg":        "#0B0F19",
    "surface":   "#131929",
    "border":    "#1E2D45",
    "primary":   "#2563EB",
    "accent":    "#F59E0B",
    "positive":  "#10B981",
    "negative":  "#EF4444",
    "text":      "#E2E8F0",
    "muted":     "#64748B",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {C['bg']} !important;
    color: {C['text']};
    font-family: 'Inter', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: {C['surface']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

/* ── KPI cards ── */
.kpi {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
}}
.kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: {C['primary']};
}}
.kpi.positive::before {{ background: {C['positive']}; }}
.kpi.negative::before {{ background: {C['negative']}; }}
.kpi.warning::before  {{ background: {C['accent']}; }}
.kpi-label {{ font-size: 0.68rem; color: {C['muted']}; text-transform: uppercase; letter-spacing: .09em; margin-bottom: 6px; }}
.kpi-value {{ font-size: 1.75rem; font-weight: 700; color: {C['text']}; line-height: 1.1; }}
.kpi-delta {{ font-size: 0.75rem; margin-top: 6px; font-weight: 500; }}
.kpi-delta.up   {{ color: {C['positive']}; }}
.kpi-delta.down {{ color: {C['negative']}; }}
.kpi-delta.flat {{ color: {C['muted']}; }}

/* ── Módulo banner ── */
.mod-banner {{
    background: linear-gradient(100deg, {C['primary']}18 0%, transparent 70%);
    border-left: 3px solid {C['primary']};
    border-radius: 0 8px 8px 0;
    padding: 12px 20px;
    margin-bottom: 24px;
}}
.mod-banner h3 {{ margin: 0 0 2px; font-size: 1.1rem; font-weight: 600; }}
.mod-banner p  {{ margin: 0; font-size: 0.78rem; color: {C['muted']}; }}

/* ── RLS badge ── */
.rls-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {C['positive']}15;
    border: 1px solid {C['positive']}40;
    color: {C['positive']};
    font-size: 0.7rem; font-weight: 500;
    border-radius: 20px; padding: 3px 10px;
}}

/* ── Insights box ── */
.insight-box {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 20px 24px;
    line-height: 1.7;
    font-size: 0.9rem;
}}
.insight-box ul {{ margin: 8px 0 0 0; padding-left: 18px; }}
.insight-box li {{ margin-bottom: 8px; }}

/* ── Code mono ── */
.code-block {{
    background: #0D1117;
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: {C['accent']};
    overflow-x: auto;
}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: {C['muted']} !important;
    font-weight: 500;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C['text']} !important;
    border-bottom-color: {C['primary']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. CONTEXTO SEMÁNTICO — diccionario de dominio para los prompts de IA     ║
# ║     Este bloque ES la "matriz de empleados" que da contexto.               ║
# ║     Nunca se envían filas de datos al modelo, solo este diccionario.       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONTEXTO_DOMINIO = {

    "empresa": """
Multinacional de servicios profesionales con presencia en 8 países:
España (ES), Portugal (PT), Colombia (CO), Argentina (AR), Uruguay (UY),
Perú (PE), Paraguay (PY) y USA. Facturación consolidada en euros.
La dirección evalúa el desempeño mensual comparando Real vs Budget y Real vs año anterior (2025).
La estacionalidad es moderada y estable entre períodos.
""",

    "financiero": {
        "descripcion": "Cuenta de resultados simplificada por país y mes. Mide la salud económica del negocio.",
        "columnas": {
            "País":           "Código ISO del país. Valores válidos: ES, PT, CO, AR, UY, PE, PY, USA.",
            "Mes":            "Período en formato YYYY-MM.",
            "Venta R26":      "Facturación real acumulada 2026 en euros. Es el ingreso por servicios facturados y reconocidos.",
            "Venta R25":      "Facturación real mismo período 2025. Sirve para comparativa interanual.",
            "Venta Budget":   "Objetivo de facturación presupuestado para 2026. Aprobado en el cierre del ejercicio anterior.",
            "Margen Bruto":   "Venta R26 menos costes directos de prestación del servicio (personal directo, subcontratas). En euros.",
            "% Margen Bruto": "Margen Bruto / Venta R26. Indica la eficiencia en la prestación del servicio. Benchmark interno: >30%.",
            "EBITA R26":      "Resultado operativo 2026 antes de intereses, impuestos y amortizaciones. En euros.",
            "EBITA R25":      "EBITA mismo período 2025.",
            "EBITA Budget":   "Objetivo de EBITA presupuestado para 2026.",
            "% EBITA":        "EBITA R26 / Venta R26. Indica rentabilidad operativa. Benchmark interno: >10%.",
        },
        "kpis_clave": [
            "Desviación Venta vs Budget (%): (Venta R26 - Venta Budget) / Venta Budget * 100",
            "Crecimiento interanual Venta (%): (Venta R26 - Venta R25) / Venta R25 * 100",
            "Desviación EBITA vs Budget (%): (EBITA R26 - EBITA Budget) / EBITA Budget * 100",
            "Expansión de margen: diferencia de % EBITA entre R26 y R25",
        ],
        "insights_interesantes": [
            "¿Qué países están por encima y por debajo del budget en venta y EBITA?",
            "¿El crecimiento interanual es real o hay efecto de tipo de cambio (países Latam)?",
            "¿El margen bruto mejora o se deteriora respecto a 2025? Indica pricing power o pérdida de eficiencia.",
            "¿Hay países con buena venta pero mal EBITA? Señal de problemas de coste no controlados.",
            "¿El grupo en su conjunto cubre el budget acumulado a este mes?",
        ],
    },

    "comercial": {
        "descripcion": "Funnel comercial por país y mes. Mide la capacidad de captación y conversión de negocio nuevo.",
        "columnas": {
            "País":               "Código ISO del país.",
            "Mes":                "Período en formato YYYY-MM.",
            "Ofertas Abiertas":   "Oportunidades activas en el pipeline. Aún no resueltas.",
            "Ofertas Ganadas":    "Propuestas cerradas con éxito. El cliente ha adjudicado el servicio.",
            "Ofertas Perdidas":   "Propuestas descartadas o adjudicadas a la competencia.",
            "Importe Pipeline":   "Valor económico total del pipeline (abiertas + ganadas + perdidas) en euros.",
            "Importe Ganado":     "Valor económico de las ofertas ganadas en euros. Este es el indicador de nueva contratación.",
            "Tasa Conversión":    "Ofertas Ganadas / (Ofertas Ganadas + Ofertas Perdidas). Excluye las abiertas aún no resueltas. En %.",
            "Objetivo Comercial": "Meta de Importe Ganado presupuestada para 2026 en euros.",
            "Importe R25":        "Importe Ganado mismo período 2025. Para comparativa interanual.",
        },
        "kpis_clave": [
            "Cobertura de pipeline: Importe Pipeline / Objetivo Comercial. Debe ser >2x para asegurar el objetivo.",
            "Tasa de conversión: benchmark interno >35%.",
            "Crecimiento de nueva contratación: (Importe Ganado - Importe R25) / Importe R25 * 100",
            "Desviación vs objetivo: (Importe Ganado - Objetivo Comercial) / Objetivo Comercial * 100",
            "Ticket medio ganado: Importe Ganado / Ofertas Ganadas",
        ],
        "insights_interesantes": [
            "¿Qué países tienen pipeline suficiente para cubrir su objetivo anual?",
            "¿La tasa de conversión mejora o empeora respecto a 2025? Indica competitividad de propuestas.",
            "¿Hay países con muchas ofertas abiertas pero baja conversión histórica? Riesgo de sobreestimación del pipeline.",
            "¿El volumen de ofertas perdidas es estructural en algún país? Puede indicar problema de precio o propuesta de valor.",
            "¿El importe ganado acumulado cubre el ritmo necesario para alcanzar el objetivo anual?",
        ],
    },

    "operaciones": {
        "descripcion": "Gestión de plantilla y absentismo por país y mes. Mide la disponibilidad real de la fuerza laboral.",
        "columnas": {
            "País":                        "Código ISO del país.",
            "Mes":                         "Período en formato YYYY-MM.",
            "Plantilla Total":             "Número de empleados activos en nómina a cierre del período.",
            "Horas Contratadas":           "Total de horas de trabajo según contrato para toda la plantilla en el período.",
            "Horas Absentismo Total":      "Suma de todas las horas de ausencia, independientemente del tipo.",
            "Horas Absentismo IT":         "Horas perdidas por Incapacidad Temporal (bajas médicas por enfermedad o accidente).",
            "Horas Absentismo Justificado":"Horas de ausencia con causa justificada no médica (permisos, citaciones, etc.).",
            "Horas Absentismo Injustificado": "Horas de ausencia sin justificación aportada. Indicador de clima laboral.",
            "% Absentismo Total":          "Horas Absentismo Total / Horas Contratadas * 100. Indicador principal de disponibilidad.",
            "% Absentismo IT":             "Horas IT / Horas Contratadas * 100. Indicador de salud de plantilla.",
            "Objetivo % Absentismo":       "Umbral máximo de absentismo total definido por la compañía. Superar este umbral activa alertas.",
            "% Absentismo R25":            "% Absentismo Total del mismo período 2025. Para tendencia interanual.",
        },
        "kpis_clave": [
            "Desviación vs objetivo: % Absentismo Total - Objetivo % Absentismo. Positivo = por encima del umbral (malo).",
            "Variación interanual: % Absentismo Total - % Absentismo R25. Indica tendencia.",
            "Peso IT sobre total: Horas IT / Horas Absentismo Total * 100. >60% indica problema de salud laboral.",
            "Horas efectivas: Horas Contratadas - Horas Absentismo Total. Disponibilidad real para facturar.",
            "Ratio horas/empleado: Horas Contratadas / Plantilla Total. Detecta cambios en tipología de contrato.",
        ],
        "insights_interesantes": [
            "¿Qué países superan el objetivo de absentismo? ¿Es puntual o acumula varios meses?",
            "¿El absentismo IT es la causa principal en los países con peor dato? Requiere plan de salud laboral.",
            "¿El absentismo injustificado es relevante en algún país? Señal de clima laboral deteriorado.",
            "¿El crecimiento de plantilla va acompañado de estabilidad en el % de absentismo?",
            "¿Hay países donde el absentismo mejora vs 2025? Identificar buenas prácticas replicables.",
        ],
    },
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. GENERACIÓN DE DATOS SIMULADOS  (@st.cache_data)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PAISES = ["ES", "PT", "CO", "AR", "UY", "PE", "PY", "USA"]
MESES  = [f"2026-{str(m).zfill(2)}" for m in range(1, 13)]

# Perfiles por país: (base_venta_M€, margen_bruto_pct, ebita_pct, plantilla_base)
PERFILES_PAIS = {
    "ES":  (8.0, 0.38, 0.14, 420),
    "PT":  (3.5, 0.35, 0.12, 180),
    "CO":  (2.8, 0.32, 0.10, 210),
    "AR":  (1.9, 0.29, 0.08, 160),
    "UY":  (1.2, 0.33, 0.11, 90),
    "PE":  (1.5, 0.31, 0.09, 120),
    "PY":  (0.8, 0.30, 0.08, 65),
    "USA": (5.5, 0.40, 0.16, 310),
}

@st.cache_data
def generar_datos() -> dict:
    """
    Simula las 3 tablas certificadas con datos coherentes para 2026.
    Incluye varianza realista, budget ligeramente optimista y R25 como base histórica.
    En producción: sustituir por lecturas a Fabric Gold / SharePoint.
    """
    rng = np.random.default_rng(42)
    filas_fin, filas_com, filas_ops = [], [], []

    for pais in PAISES:
        base_v, mb_pct, ebt_pct, plantilla = PERFILES_PAIS[pais]

        for i, mes in enumerate(MESES):
            # Factor estacional suave
            estacional = 1 + 0.04 * np.sin(i * np.pi / 6)

            # ── FINANCIERO ───────────────────────────────────────────────────
            venta_r26    = base_v * 1e6 * estacional * rng.uniform(0.88, 1.12)
            venta_r25    = venta_r26 * rng.uniform(0.88, 1.05)
            venta_budget = venta_r26 * rng.uniform(0.95, 1.10)
            mb           = venta_r26 * mb_pct * rng.uniform(0.93, 1.07)
            mb_pct_real  = mb / venta_r26 * 100
            ebita_r26    = venta_r26 * ebt_pct * rng.uniform(0.85, 1.15)
            ebita_r25    = ebita_r26 * rng.uniform(0.87, 1.06)
            ebita_budget = ebita_r26 * rng.uniform(0.96, 1.12)
            pct_ebita    = ebita_r26 / venta_r26 * 100

            filas_fin.append({
                "País": pais, "Mes": mes,
                "Venta R26": round(venta_r26, 0),
                "Venta R25": round(venta_r25, 0),
                "Venta Budget": round(venta_budget, 0),
                "Margen Bruto": round(mb, 0),
                "% Margen Bruto": round(mb_pct_real, 2),
                "EBITA R26": round(ebita_r26, 0),
                "EBITA R25": round(ebita_r25, 0),
                "EBITA Budget": round(ebita_budget, 0),
                "% EBITA": round(pct_ebita, 2),
            })

            # ── COMERCIAL ────────────────────────────────────────────────────
            ganadas  = int(rng.integers(8, 35))
            perdidas = int(rng.integers(5, 25))
            abiertas = int(rng.integers(15, 60))
            ticket   = base_v * 1e6 / 12 / 20 * rng.uniform(0.7, 1.4)
            imp_gan  = ganadas * ticket
            imp_r25  = imp_gan * rng.uniform(0.85, 1.08)
            obj_com  = imp_gan * rng.uniform(0.92, 1.15)
            tasa_cv  = ganadas / (ganadas + perdidas) * 100
            imp_pip  = (ganadas + perdidas + abiertas) * ticket

            filas_com.append({
                "País": pais, "Mes": mes,
                "Ofertas Abiertas":   abiertas,
                "Ofertas Ganadas":    ganadas,
                "Ofertas Perdidas":   perdidas,
                "Importe Pipeline":   round(imp_pip, 0),
                "Importe Ganado":     round(imp_gan, 0),
                "Tasa Conversión":    round(tasa_cv, 2),
                "Objetivo Comercial": round(obj_com, 0),
                "Importe R25":        round(imp_r25, 0),
            })

            # ── OPERACIONES ──────────────────────────────────────────────────
            plant   = int(plantilla * rng.uniform(0.95, 1.08))
            h_cont  = plant * 168  # ~168h/mes por empleado
            pct_it  = rng.uniform(1.5, 5.5)
            pct_jus = rng.uniform(0.3, 1.5)
            pct_inj = rng.uniform(0.0, 0.8)
            pct_tot = pct_it + pct_jus + pct_inj
            h_it    = int(h_cont * pct_it / 100)
            h_jus   = int(h_cont * pct_jus / 100)
            h_inj   = int(h_cont * pct_inj / 100)
            h_tot   = h_it + h_jus + h_inj
            obj_abs = round(rng.uniform(3.5, 5.0), 1)
            pct_r25 = round(pct_tot * rng.uniform(0.88, 1.12), 2)

            filas_ops.append({
                "País": pais, "Mes": mes,
                "Plantilla Total":               plant,
                "Horas Contratadas":             h_cont,
                "Horas Absentismo Total":        h_tot,
                "Horas Absentismo IT":           h_it,
                "Horas Absentismo Justificado":  h_jus,
                "Horas Absentismo Injustificado":h_inj,
                "% Absentismo Total":            round(pct_tot, 2),
                "% Absentismo IT":               round(pct_it, 2),
                "Objetivo % Absentismo":         obj_abs,
                "% Absentismo R25":              pct_r25,
            })

    return {
        "financiero":  pd.DataFrame(filas_fin),
        "comercial":   pd.DataFrame(filas_com),
        "operaciones": pd.DataFrame(filas_ops),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. SEGURIDAD A NIVEL DE FILA  (Row-Level Security)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

RLS_MAP = {
    "manager_latam@empresa.com":  ["CO", "AR", "UY", "PE", "PY"],
    "manager_iberia@empresa.com": ["ES", "PT"],
    "global_admin@empresa.com":   PAISES,
    "analyst_usa@empresa.com":    ["USA"],
}
USUARIO_ACTIVO = "manager_latam@empresa.com"


def aplicar_rls(tablas_raw: dict, usuario: str) -> tuple[dict, list]:
    """
    ★ PUNTO DE CONTROL RLS ★
    Filtra las 3 tablas antes de cualquier otra lógica.
    Deny-by-default: usuario no mapeado → DataFrames vacíos.
    """
    paises_ok = RLS_MAP.get(usuario, [])
    return (
        {k: df[df["País"].isin(paises_ok)].copy() for k, df in tablas_raw.items()},
        paises_ok,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. AGENTE ENRUTADOR                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

RUTAS = {
    "financiero":  ["venta", "ventas", "ebita", "financiero", "ingreso", "margen",
                    "facturación", "facturacion", "budget", "revenue", "resultado"],
    "comercial":   ["oferta", "comercial", "pipeline", "propuesta", "importe ganado",
                    "captación", "captacion", "conversión", "conversion", "contratación"],
    "operaciones": ["absentismo", "operaciones", "horas", "ausencia", "plantilla",
                    "it ", "baja", "incapacidad", "workforce", "empleado"],
}

def enrutar(texto: str) -> str | None:
    t = texto.lower()
    for modulo, kws in RUTAS.items():
        if any(kw in t for kw in kws):
            return modulo
    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. LLAMADAS A CLAUDE API                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if CLAUDE_DISPONIBLE and api_key:
        return anthropic.Anthropic(api_key=api_key)
    return None


def generar_codigo_plotly(prompt_usuario: str, modulo: str, columnas: list[str], df: pd.DataFrame) -> str:
    """
    ★ INTEGRACIÓN CLAUDE → VISUAL IA ★
    Envía solo cabeceras + estadísticas agregadas (nunca filas individuales).
    Devuelve una línea de código Plotly Express lista para exec().
    """
    ctx_modulo = CONTEXTO_DOMINIO[modulo]
    stats = df.describe().to_string()

    system = f"""
Eres un experto en visualización de datos con Plotly Express para una audiencia directiva.
Contexto del módulo '{modulo}': {ctx_modulo['descripcion']}

Columnas disponibles y su significado:
{json.dumps(ctx_modulo['columnas'], ensure_ascii=False, indent=2)}

KPIs clave del módulo:
{chr(10).join(ctx_modulo['kpis_clave'])}

REGLAS ESTRICTAS:
- Responde ÚNICAMENTE con una línea de código Python válida.
- Usa SIEMPRE 'df_final' como nombre del DataFrame.
- Usa SOLO estas columnas exactas: {columnas}
- No añadas imports, comentarios, ni llames a .show().
- Usa template='plotly_dark' en todos los gráficos.
- La línea debe empezar por px. (bar, line, scatter, pie, box, area, choropleth...)
- Elige el tipo de gráfico más apropiado para la pregunta del usuario.
- Añade siempre un title= descriptivo y en español.
""".strip()

    cliente = _get_client()
    if cliente:
        # ── LLAMADA REAL A CLAUDE ────────────────────────────────────────────
        msg = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": prompt_usuario}],
        )
        return msg.content[0].text.strip()
    else:
        # ── STUB DETERMINISTA (desarrollo sin API key) ───────────────────────
        metricas = [c for c in columnas if c not in ("País", "Mes")]
        col_y    = metricas[0] if metricas else columnas[-1]
        tipo     = "bar"
        color    = ""
        if any(w in prompt_usuario.lower() for w in ["línea", "linea", "tendencia", "evolución", "evolucion", "tiempo"]):
            tipo  = "line"
            color = ", color='País'"
        elif any(w in prompt_usuario.lower() for w in ["dispersión", "scatter", "correlación", "correlacion"]):
            tipo  = "scatter"
            col_y2 = metricas[1] if len(metricas) > 1 else col_y
            return (f"px.scatter(df_final, x='{col_y}', y='{col_y2}', color='País', "
                    f"size='{col_y}', template='plotly_dark', title='Correlación {col_y} vs {col_y2}')")
        elif any(w in prompt_usuario.lower() for w in ["pastel", "pie", "distribución", "distribucion"]):
            return (f"px.pie(df_final.groupby('País')['{col_y}'].sum().reset_index(), "
                    f"values='{col_y}', names='País', template='plotly_dark', title='Distribución de {col_y} por país')")
        return (f"px.{tipo}(df_final.groupby('Mes')['{col_y}'].sum().reset_index(), "
                f"x='Mes', y='{col_y}'{color}, template='plotly_dark', title='{col_y} por mes')")


def generar_insights(modulo: str, df: pd.DataFrame) -> str:
    """
    ★ INTEGRACIÓN CLAUDE → INSIGHTS C-LEVEL ★
    Envía estadísticas agregadas + contexto semántico completo.
    Nunca envía filas individuales al modelo.
    Devuelve HTML con bullets de insights ejecutivos.
    """
    ctx = CONTEXTO_DOMINIO[modulo]
    empresa_ctx = CONTEXTO_DOMINIO["empresa"]

    # Construir resumen estadístico agregado (sin filas individuales)
    df_agg = df.groupby("País").mean(numeric_only=True).round(2)
    resumen_stats = df_agg.to_string()

    # Totales del período
    totales = df.select_dtypes(include="number").sum().round(0).to_dict()
    medias  = df.select_dtypes(include="number").mean().round(2).to_dict()

    system = f"""
Eres un analista financiero senior que prepara un briefing ejecutivo para el C-Level de una multinacional.
Tu objetivo es identificar los hallazgos más relevantes y accionables, NO describir los datos.

Contexto empresa: {empresa_ctx}
Módulo analizado: {ctx['descripcion']}

Definición de columnas:
{json.dumps(ctx['columnas'], ensure_ascii=False, indent=2)}

KPIs clave a evaluar:
{chr(10).join(ctx['kpis_clave'])}

Preguntas que interesan a la dirección:
{chr(10).join(ctx['insights_interesantes'])}

REGLAS DE REDACCIÓN:
- Máximo 5 bullets. Cada bullet = 1 insight accionable.
- Tono ejecutivo: directo, sin tecnicismos, orientado a decisión.
- Nombra países específicos cuando el dato lo justifique.
- Indica si un dato es positivo ✅ o requiere atención ⚠️ o es crítico 🔴.
- Responde en español.
- Formato: lista HTML con <li> dentro de <ul>. Sin <html>, <body> ni otros tags externos.
""".strip()

    user_msg = f"""
Datos del período analizado (medias por país):
{resumen_stats}

Totales del período:
{json.dumps({k: float(v) for k, v in totales.items()}, ensure_ascii=False)}

Genera los insights ejecutivos.
"""

    cliente = _get_client()
    if cliente:
        msg = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return msg.content[0].text.strip()
    else:
        # ── STUB insights ────────────────────────────────────────────────────
        stubs = {
            "financiero": [
                "✅ La venta consolidada del grupo supera el presupuesto en el período analizado, con Colombia y Uruguay como principales contribuidores positivos.",
                "⚠️ Argentina y Perú presentan desviaciones negativas en EBITA respecto al budget — requieren revisión de estructura de costes.",
                "✅ El margen bruto del grupo se sitúa por encima del benchmark interno del 30%, con USA liderando la rentabilidad.",
                "⚠️ El crecimiento interanual de venta es positivo pero el % EBITA se mantiene estable, indicando que el crecimiento no genera apalancamiento operativo.",
                "🔴 Paraguay acumula tres meses consecutivos por debajo del budget en venta — se recomienda revisión del plan comercial local.",
            ],
            "comercial": [
                "✅ El pipeline total del grupo cubre más de 2x el objetivo anual, lo que proporciona visibilidad suficiente para el ejercicio.",
                "⚠️ La tasa de conversión en Argentina y Perú está por debajo del benchmark interno del 35% — revisar calidad de propuestas.",
                "✅ Colombia lidera la nueva contratación con crecimiento positivo vs 2025, impulsado por aumento de ticket medio.",
                "⚠️ El volumen de ofertas perdidas en Uruguay es estructuralmente alto — posible problema de competitividad en precio.",
                "🔴 Paraguay muestra cobertura de pipeline insuficiente respecto a su objetivo anual — riesgo de incumplimiento en H2.",
            ],
            "operaciones": [
                "⚠️ Tres países superan el objetivo de absentismo definido por la compañía, con el absentismo IT como causa principal.",
                "✅ Colombia mantiene el % de absentismo estable pese al crecimiento de plantilla, indicando buena integración de nuevas incorporaciones.",
                "🔴 Argentina presenta el mayor incremento de absentismo IT vs 2025 — se recomienda intervención del programa de salud laboral.",
                "⚠️ El absentismo injustificado en Paraguay, aunque bajo en valor absoluto, ha aumentado respecto al año anterior — señal de clima laboral a monitorizar.",
                "✅ Uruguay y Perú muestran mejora interanual en % de absentismo total — identificar y replicar las buenas prácticas aplicadas.",
            ],
        }
        items = "".join(f"<li>{s}</li>" for s in stubs.get(modulo, ["No hay insights disponibles."]))
        return f"<ul>{items}</ul>"


def ejecutar_plotly(codigo: str, df_final: pd.DataFrame):
    """Ejecuta el código Plotly generado en un namespace aislado."""
    ns = {"px": px, "go": go, "pd": pd, "df_final": df_final}
    try:
        exec(f"__fig__ = {codigo}", ns)
        return ns.get("__fig__")
    except Exception as e:
        st.error(f"Error al renderizar la visualización: `{e}`")
        st.markdown(f'<div class="code-block">{codigo}</div>', unsafe_allow_html=True)
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. COMPONENTES DE UI                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fmt_M(v: float) -> str:
    if abs(v) >= 1_000_000: return f"€{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"€{v/1_000:.0f}K"
    return f"€{v:.0f}"

def fmt_pct(v: float) -> str: return f"{v:.1f}%"

def delta_class(v: float) -> str:
    return "up" if v > 0 else ("down" if v < 0 else "flat")

def delta_icon(v: float) -> str:
    return "▲" if v > 0 else ("▼" if v < 0 else "–")

def kpi(label: str, value: str, delta_txt: str = "", clase: str = ""):
    delta_html = f'<div class="kpi-delta {delta_class(0) if not delta_txt else ""}">{delta_txt}</div>' if delta_txt else ""
    if delta_txt:
        sign = 1 if "▲" in delta_txt or "+" in delta_txt else -1
        delta_html = f'<div class="kpi-delta {delta_class(sign)}">{delta_txt}</div>'
    st.markdown(
        f'<div class="kpi {clase}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )

def banner(titulo: str, desc: str = ""):
    extra = f'<p>{desc}</p>' if desc else ""
    st.markdown(f'<div class="mod-banner"><h3>{titulo}</h3>{extra}</div>', unsafe_allow_html=True)

def pt() -> dict:
    """Tema Plotly unificado."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter"),
        margin=dict(t=40, b=30, l=10, r=10),
    )

def seccion_insights(modulo: str, df: pd.DataFrame):
    """Panel de insights IA con botón de regeneración."""
    st.markdown("---")
    st.markdown("### 💡 Insights ejecutivos")
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        generar = st.button("Generar insights IA", key=f"btn_insights_{modulo}", type="primary")
    with col_info:
        if not _get_client():
            st.caption("⚠️ Modo demo — configura ANTHROPIC_API_KEY para insights reales.")

    key_cache = f"insights_{modulo}"
    if generar or key_cache in st.session_state:
        if generar:
            with st.spinner("Analizando datos con IA..."):
                st.session_state[key_cache] = generar_insights(modulo, df)
        html = st.session_state.get(key_cache, "")
        if html:
            st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)

def seccion_visual_ia(modulo: str, df: pd.DataFrame):
    """Panel de generación de visualizaciones opacas."""
    st.markdown("---")
    st.markdown("### 🎨 Crear mi propia visualización")
    st.caption("Describe en lenguaje natural qué quieres ver. La IA generará el gráfico automáticamente.")

    prompt_viz = st.text_input(
        "¿Qué quieres visualizar?",
        placeholder="Ej: evolución mensual de las ventas por país, dispersión entre margen y EBITA...",
        key=f"viz_prompt_{modulo}",
        label_visibility="collapsed",
    )

    if st.button("Generar visualización", key=f"btn_viz_{modulo}"):
        if prompt_viz.strip():
            with st.spinner("Generando visualización..."):
                codigo = generar_codigo_plotly(prompt_viz, modulo, list(df.columns), df)
            # Mostrar código generado en modo colapsado (el proceso es opaco por defecto)
            with st.expander("Ver código generado", expanded=False):
                st.markdown(f'<div class="code-block">{codigo}</div>', unsafe_allow_html=True)
            fig = ejecutar_plotly(codigo, df)
            if fig:
                fig.update_layout(**pt())
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Escribe una descripción para generar la visualización.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. MÓDULOS DE BI ESTÁNDAR                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_financiero(df: pd.DataFrame):
    banner("📈 Módulo Financiero", "Venta, Margen Bruto y EBITA · Real vs Budget vs 2025")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    venta_tot   = df["Venta R26"].sum()
    venta_bud   = df["Venta Budget"].sum()
    venta_r25   = df["Venta R25"].sum()
    ebita_tot   = df["EBITA R26"].sum()
    ebita_bud   = df["EBITA Budget"].sum()
    mb_tot      = df["Margen Bruto"].sum()
    pct_mb      = mb_tot / venta_tot * 100
    pct_ebita   = ebita_tot / venta_tot * 100
    dev_venta   = (venta_tot - venta_bud) / venta_bud * 100
    dev_ebita   = (ebita_tot - ebita_bud) / ebita_bud * 100
    crec_venta  = (venta_tot - venta_r25) / venta_r25 * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Venta Real 2026",  fmt_M(venta_tot),
                 f"{delta_icon(dev_venta)} {dev_venta:+.1f}% vs Budget",
                 "positive" if dev_venta >= 0 else "negative")
    with c2: kpi("EBITA Real 2026",  fmt_M(ebita_tot),
                 f"{delta_icon(dev_ebita)} {dev_ebita:+.1f}% vs Budget",
                 "positive" if dev_ebita >= 0 else "negative")
    with c3: kpi("% Margen Bruto",   fmt_pct(pct_mb),
                 f"Benchmark interno: >30%",
                 "positive" if pct_mb >= 30 else "warning")
    with c4: kpi("% EBITA",          fmt_pct(pct_ebita),
                 f"{delta_icon(crec_venta)} Venta {crec_venta:+.1f}% vs 2025",
                 "positive" if pct_ebita >= 10 else "warning")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficos BI estándar ──────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["Venta", "EBITA & Margen", "Ranking países"])

    with t1:
        df_mes = df.groupby("Mes")[["Venta R26", "Venta Budget", "Venta R25"]].sum().reset_index()
        fig = go.Figure()
        fig.add_bar(x=df_mes["Mes"], y=df_mes["Venta Budget"], name="Budget",
                    marker_color=C["muted"], opacity=0.5)
        fig.add_bar(x=df_mes["Mes"], y=df_mes["Venta R25"],    name="Real 2025",
                    marker_color=C["accent"], opacity=0.7)
        fig.add_bar(x=df_mes["Mes"], y=df_mes["Venta R26"],    name="Real 2026",
                    marker_color=C["primary"])
        fig.update_layout(barmode="group", title="Venta mensual: Real vs Budget vs 2025", **pt())
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="Mes", y="Venta R26", color="País",
                       title="Evolución de Venta por País")
        fig2.update_layout(**pt())
        st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_mes2 = df.groupby("Mes")[["EBITA R26", "EBITA Budget", "Margen Bruto"]].sum().reset_index()
        fig3 = go.Figure()
        fig3.add_bar(x=df_mes2["Mes"], y=df_mes2["EBITA Budget"], name="EBITA Budget",
                     marker_color=C["muted"], opacity=0.5)
        fig3.add_bar(x=df_mes2["Mes"], y=df_mes2["EBITA R26"],    name="EBITA Real 2026",
                     marker_color=C["positive"])
        fig3.update_layout(barmode="group", title="EBITA mensual: Real vs Budget", **pt())
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.area(df.groupby("Mes")[["% Margen Bruto", "% EBITA"]].mean().reset_index(),
                       x="Mes", y=["% Margen Bruto", "% EBITA"],
                       title="Evolución de Márgenes (%)", template="plotly_dark")
        fig4.update_layout(**pt())
        st.plotly_chart(fig4, use_container_width=True)

    with t3:
        df_pais = df.groupby("País").agg(
            Venta=("Venta R26", "sum"),
            Budget=("Venta Budget", "sum"),
            EBITA=("EBITA R26", "sum"),
        ).reset_index()
        df_pais["Desv Budget %"] = (df_pais["Venta"] - df_pais["Budget"]) / df_pais["Budget"] * 100
        df_pais = df_pais.sort_values("Desv Budget %", ascending=False)
        fig5 = px.bar(df_pais, x="País", y="Desv Budget %",
                      color="Desv Budget %",
                      color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                      title="Desviación Venta vs Budget por País (%)",
                      template="plotly_dark")
        fig5.add_hline(y=0, line_dash="dash", line_color=C["muted"])
        fig5.update_layout(**pt())
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Tabla de datos"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    seccion_insights("financiero", df)
    seccion_visual_ia("financiero", df)


def render_comercial(df: pd.DataFrame):
    banner("🤝 Módulo Comercial", "Funnel comercial · Ofertas, conversión y nueva contratación")

    gan_tot    = df["Ofertas Ganadas"].sum()
    per_tot    = df["Ofertas Perdidas"].sum()
    tasa_cv    = gan_tot / (gan_tot + per_tot) * 100 if (gan_tot + per_tot) > 0 else 0
    imp_gan    = df["Importe Ganado"].sum()
    obj_com    = df["Objetivo Comercial"].sum()
    imp_r25    = df["Importe R25"].sum()
    dev_obj    = (imp_gan - obj_com) / obj_com * 100
    crec_com   = (imp_gan - imp_r25) / imp_r25 * 100
    pip_tot    = df["Importe Pipeline"].sum()
    cobertura  = pip_tot / obj_com if obj_com > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Importe Ganado", fmt_M(imp_gan),
                 f"{delta_icon(dev_obj)} {dev_obj:+.1f}% vs Objetivo",
                 "positive" if dev_obj >= 0 else "negative")
    with c2: kpi("Tasa Conversión", fmt_pct(tasa_cv),
                 "Benchmark: >35%",
                 "positive" if tasa_cv >= 35 else "warning")
    with c3: kpi("Cobertura Pipeline", f"{cobertura:.1f}x",
                 "Recomendado: >2x",
                 "positive" if cobertura >= 2 else "warning")
    with c4: kpi("Crecimiento vs 2025", fmt_pct(crec_com),
                 f"{delta_icon(crec_com)} interanual",
                 "positive" if crec_com >= 0 else "negative")

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Funnel", "Conversión", "Ranking países"])

    with t1:
        df_mes = df.groupby("Mes")[["Ofertas Abiertas", "Ofertas Ganadas", "Ofertas Perdidas"]].sum().reset_index()
        fig = px.bar(df_mes, x="Mes",
                     y=["Ofertas Ganadas", "Ofertas Abiertas", "Ofertas Perdidas"],
                     title="Funnel comercial mensual",
                     color_discrete_map={"Ofertas Ganadas": C["positive"],
                                         "Ofertas Abiertas": C["primary"],
                                         "Ofertas Perdidas": C["negative"]},
                     template="plotly_dark")
        fig.update_layout(**pt())
        st.plotly_chart(fig, use_container_width=True)

        df_imp = df.groupby("Mes")[["Importe Ganado", "Objetivo Comercial", "Importe R25"]].sum().reset_index()
        fig2 = go.Figure()
        fig2.add_bar(x=df_imp["Mes"], y=df_imp["Objetivo Comercial"], name="Objetivo",
                     marker_color=C["muted"], opacity=0.5)
        fig2.add_bar(x=df_imp["Mes"], y=df_imp["Importe R25"], name="Real 2025",
                     marker_color=C["accent"], opacity=0.7)
        fig2.add_bar(x=df_imp["Mes"], y=df_imp["Importe Ganado"], name="Ganado 2026",
                     marker_color=C["positive"])
        fig2.update_layout(barmode="group", title="Importe Ganado vs Objetivo vs 2025", **pt())
        st.plotly_chart(fig2, use_container_width=True)

    with t2:
        fig3 = px.line(df.groupby("Mes")["Tasa Conversión"].mean().reset_index(),
                       x="Mes", y="Tasa Conversión",
                       title="Evolución Tasa de Conversión (%)", template="plotly_dark")
        fig3.add_hline(y=35, line_dash="dash", line_color=C["accent"],
                       annotation_text="Benchmark 35%")
        fig3.update_layout(**pt())
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(df, x="Ofertas Ganadas", y="Importe Ganado", color="País",
                          size="Tasa Conversión",
                          title="Volumen vs Importe vs Conversión por País",
                          template="plotly_dark")
        fig4.update_layout(**pt())
        st.plotly_chart(fig4, use_container_width=True)

    with t3:
        df_p = df.groupby("País").agg(
            Ganado=("Importe Ganado", "sum"),
            Objetivo=("Objetivo Comercial", "sum"),
            Tasa=("Tasa Conversión", "mean"),
        ).reset_index()
        df_p["Desv %"] = (df_p["Ganado"] - df_p["Objetivo"]) / df_p["Objetivo"] * 100
        df_p = df_p.sort_values("Desv %", ascending=False)
        fig5 = px.bar(df_p, x="País", y="Desv %",
                      color="Desv %",
                      color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                      title="Desviación Importe Ganado vs Objetivo por País (%)",
                      template="plotly_dark")
        fig5.add_hline(y=0, line_dash="dash", line_color=C["muted"])
        fig5.update_layout(**pt())
        st.plotly_chart(fig5, use_container_width=True)

    with st.expander("📋 Tabla de datos"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    seccion_insights("comercial", df)
    seccion_visual_ia("comercial", df)


def render_operaciones(df: pd.DataFrame):
    banner("⚙️ Módulo Operaciones", "Absentismo y disponibilidad de plantilla · Real vs Objetivo vs 2025")

    plant_tot  = df.groupby("País")["Plantilla Total"].mean().sum()
    h_cont     = df["Horas Contratadas"].sum()
    h_abs      = df["Horas Absentismo Total"].sum()
    pct_abs    = h_abs / h_cont * 100
    pct_obj    = df["Objetivo % Absentismo"].mean()
    pct_r25    = df["% Absentismo R25"].mean()
    pct_it_med = df["% Absentismo IT"].mean()
    var_r25    = pct_abs - pct_r25
    dev_obj    = pct_abs - pct_obj

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Plantilla Media", f"{plant_tot:,.0f} emp.", "", "")
    with c2: kpi("% Absentismo Total", fmt_pct(pct_abs),
                 f"{delta_icon(-dev_obj)} {dev_obj:+.1f}pp vs Objetivo",
                 "positive" if dev_obj <= 0 else "negative")
    with c3: kpi("% Absentismo IT", fmt_pct(pct_it_med),
                 "Componente médico",
                 "warning" if pct_it_med > 3 else "positive")
    with c4: kpi("Variación vs 2025", f"{var_r25:+.2f}pp",
                 "Mejora ▲ = más bajo",
                 "positive" if var_r25 <= 0 else "negative")

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Evolución", "Desglose", "Ranking países"])

    with t1:
        df_mes = df.groupby("Mes")[["% Absentismo Total", "% Absentismo R25", "Objetivo % Absentismo"]].mean().reset_index()
        fig = go.Figure()
        fig.add_scatter(x=df_mes["Mes"], y=df_mes["Objetivo % Absentismo"],
                        name="Objetivo", line=dict(color=C["accent"], dash="dash"))
        fig.add_scatter(x=df_mes["Mes"], y=df_mes["% Absentismo R25"],
                        name="Real 2025", line=dict(color=C["muted"]))
        fig.add_scatter(x=df_mes["Mes"], y=df_mes["% Absentismo Total"],
                        name="Real 2026", line=dict(color=C["primary"], width=2.5),
                        fill="tonexty", fillcolor=f"{C['primary']}20")
        fig.update_layout(title="% Absentismo mensual: Real vs Objetivo vs 2025", **pt())
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="Mes", y="% Absentismo Total", color="País",
                       title="% Absentismo por País", template="plotly_dark")
        fig2.add_hline(y=pct_obj, line_dash="dash", line_color=C["accent"],
                       annotation_text="Objetivo")
        fig2.update_layout(**pt())
        st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_tipo = df[["Mes", "Horas Absentismo IT",
                       "Horas Absentismo Justificado",
                       "Horas Absentismo Injustificado"]].groupby("Mes").sum().reset_index()
        fig3 = px.bar(df_tipo, x="Mes",
                      y=["Horas Absentismo IT", "Horas Absentismo Justificado", "Horas Absentismo Injustificado"],
                      title="Desglose de horas de absentismo por tipo",
                      color_discrete_map={
                          "Horas Absentismo IT":            C["negative"],
                          "Horas Absentismo Justificado":   C["accent"],
                          "Horas Absentismo Injustificado": C["muted"],
                      },
                      template="plotly_dark")
        fig3.update_layout(**pt())
        st.plotly_chart(fig3, use_container_width=True)

        df_plant = df.groupby("Mes")[["Plantilla Total"]].mean().reset_index()
        fig4 = px.line(df_plant, x="Mes", y="Plantilla Total",
                       title="Evolución de Plantilla Media", template="plotly_dark")
        fig4.update_layout(**pt())
        st.plotly_chart(fig4, use_container_width=True)

    with t3:
        df_p = df.groupby("País").agg(
            Abs=("% Absentismo Total", "mean"),
            Obj=("Objetivo % Absentismo", "mean"),
            R25=("% Absentismo R25", "mean"),
        ).reset_index()
        df_p["Desv Objetivo"] = df_p["Abs"] - df_p["Obj"]
        df_p = df_p.sort_values("Desv Objetivo", ascending=False)
        fig5 = px.bar(df_p, x="País", y="Desv Objetivo",
                      color="Desv Objetivo",
                      color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                      title="Desviación % Absentismo vs Objetivo por País (pp)",
                      template="plotly_dark")
        fig5.add_hline(y=0, line_dash="dash", line_color=C["muted"])
        fig5.update_layout(**pt())
        st.plotly_chart(fig5, use_container_width=True)

    with st.expander("📋 Tabla de datos"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    seccion_insights("operaciones", df)
    seccion_visual_ia("operaciones", df)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. AGENTE DE CONSULTA (chat + enrutador)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

LABELS_MODULO = {
    "financiero":  "📈 Módulo Financiero",
    "comercial":   "🤝 Módulo Comercial",
    "operaciones": "⚙️ Módulo Operaciones",
}

def render_agente(tablas: dict):
    st.title("🤖 Agente de Consulta")
    st.markdown("Escribe qué quieres analizar y el agente identificará el módulo correcto.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "modulo_agente" not in st.session_state:
        st.session_state.modulo_agente = None
    if "prompt_agente" not in st.session_state:
        st.session_state.prompt_agente = ""

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ej: 'Muéstrame la evolución de ventas', 'analiza el pipeline comercial'...")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        modulo = enrutar(prompt)
        st.session_state.prompt_agente = prompt

        if modulo:
            st.session_state.modulo_agente = modulo
            label = LABELS_MODULO[modulo]
            resp = (f"He identificado tu consulta como **{label}**. "
                    f"Cargando el módulo con los datos de tus países autorizados 👇")
        else:
            resp = ("No he detectado el módulo. Prueba con palabras como: "
                    "*ventas, ebita, margen, oferta, pipeline, conversión, absentismo, plantilla...*")
            st.session_state.modulo_agente = None

        st.session_state.chat.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)

    if st.session_state.modulo_agente:
        st.divider()
        mod = st.session_state.modulo_agente
        df  = tablas[mod]
        if mod == "financiero":  render_financiero(df)
        elif mod == "comercial": render_comercial(df)
        else:                    render_operaciones(df)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10. MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # ── Cargar y proteger datos con RLS ──────────────────────────────────────
    tablas_raw = generar_datos()

    # ★★★ RLS: primer y único punto de acceso a los datos ★★★
    tablas, paises_ok = aplicar_rls(tablas_raw, USUARIO_ACTIVO)
    paises_str = " · ".join(paises_ok)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏢 BI Portal")
        st.markdown("*Multinacional · 2026*")
        st.divider()
        st.markdown(f"**Usuario:** `{USUARIO_ACTIVO}`")
        st.markdown(f'<div class="rls-pill">🔒 {paises_str}</div>', unsafe_allow_html=True)
        st.divider()

        seccion = st.radio(
            "Módulo",
            ["🤖 Agente de Consulta", "📊 Cuadro de Mando Fijo"],
            label_visibility="collapsed",
        )
        st.divider()

        if not _get_client():
            st.caption("⚠️ **Modo demo**: sin ANTHROPIC_API_KEY los gráficos IA y los insights usan stubs deterministas.")
        else:
            st.caption("✅ Claude API conectado")

        st.caption("v2.0 · Datos simulados")

    # ── Routing principal ─────────────────────────────────────────────────────
    if seccion == "🤖 Agente de Consulta":
        render_agente(tablas)

    else:
        st.title("📊 Cuadro de Mando Fijo")
        st.markdown(f'<div class="rls-pill" style="margin-bottom:20px">🔒 Datos de: {paises_str}</div>',
                    unsafe_allow_html=True)

        # Filtros globales (sobre datos ya protegidos por RLS)
        with st.expander("🔧 Filtros", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                paises_sel = st.multiselect("País", paises_ok, default=paises_ok)
            with c2:
                meses_sel  = st.multiselect("Mes",  MESES,     default=MESES)

        mask = lambda df: df[df["País"].isin(paises_sel) & df["Mes"].isin(meses_sel)]
        df_fin = mask(tablas["financiero"])
        df_com = mask(tablas["comercial"])
        df_ops = mask(tablas["operaciones"])

        tab_f, tab_c, tab_o = st.tabs(["📈 Financiero", "🤝 Comercial", "⚙️ Operaciones"])
        with tab_f: render_financiero(df_fin)
        with tab_c: render_comercial(df_com)
        with tab_o: render_operaciones(df_ops)


if __name__ == "__main__":
    main()
