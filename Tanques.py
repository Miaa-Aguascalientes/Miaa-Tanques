"""Tablero de Monitoreo de Tanques de Regulación - MIAA (Versión con Nuevo Estilo CSS)"""

from dataclasses import dataclass
from html import escape
import logging

import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import plotly.graph_objects as go

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================

TIMEZONE = "America/Mexico_City"
DATA_KEY = "tank_monitoring_data"
DEFAULT_LOAD_DAYS = 30
MAX_QUERY_DAYS = 90
CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 8
CATEGORY_COLUMN = "Distrito"
MAX_AGE_MINUTES = 120
EXPECTED_INTERVAL_MINUTES = 5
MIN_COMMUNICATION_PCT = 5

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class MySQLConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 3306

    @classmethod
    def from_secrets(cls, secret_key: str) -> "MySQLConfig":
        if secret_key not in st.secrets:
            raise KeyError(f"No se encontró la sección [{secret_key}] en st.secrets.")
        sec = st.secrets[secret_key]
        return cls(
            host=sec["host"],
            database=sec["database"],
            user=sec["user"],
            password=sec["password"],
            port=int(sec.get("port", 3306)),
        )


# =====================================================================
# CAPA DE ACCESO A DATOS (Data Access Layer)
# =====================================================================


def fetch_mysql_table(
    config: MySQLConfig, table: str, columns: list[str] | None = None
) -> pd.DataFrame:
    cols = ", ".join([f"`{c}`" for c in columns]) if columns else "*"
    query = f"SELECT {cols} FROM `{table}`;"
    connection = pymysql.connect(
        host=config.host,
        user=config.user,
        password=config.password,
        database=config.database,
        port=config.port,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        return pd.DataFrame(result)
    finally:
        connection.close()


def fetch_tank_readings(
    config: MySQLConfig,
    scada_tags: dict[str, str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not scada_tags:
        return pd.DataFrame(columns=["timestamp", "tank_id", "level_m"])

    tag_to_tank = {v: k for k, v in scada_tags.items()}
    tags_list = list(scada_tags.values())
    tags_string = "','".join(tags_list)

    connection = pymysql.connect(
        host=config.host,
        user=config.user,
        password=config.password,
        database=config.database,
        port=config.port,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        query = f"""
            SELECT 
                h.TIME AS raw_time,
                t.NAME AS tag_name,
                h.VALUE AS level_m
            FROM vfitagnumhistory h
            JOIN VfiTagRef t ON h.GATEID = t.GATEID
            WHERE t.NAME IN ('{tags_string}')
              AND h.TIME >= {start_ts}
              AND h.TIME <= {end_ts}
            ORDER BY h.TIME ASC;
        """

        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()

        if not result:
            return pd.DataFrame(columns=["timestamp", "tank_id", "level_m"])

        df = pd.DataFrame(result)
        df["timestamp"] = pd.to_datetime(df["raw_time"], unit="ms", errors="coerce")
        df["tank_id"] = df["tag_name"].map(tag_to_tank)

        return df[["timestamp", "tank_id", "level_m"]].dropna(subset=["tank_id"])

    finally:
        connection.close()


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def get_tank_level_data(
    start_date: pd.Timestamp, end_date: pd.Timestamp, ntanks: int | None = None
):
    hyd_db_config = MySQLConfig.from_secrets("HYD_DB")
    scada_config = MySQLConfig.from_secrets("SCADA_DB")

    df = fetch_mysql_table(config=hyd_db_config, table="Diccionario_de_tanques")
    df_selected = df.copy()

    if ntanks is not None:
        if ntanks < 1:
            raise ValueError("ntanks debe ser mayor a cero.")
        df_selected = df.head(ntanks).copy()

    if df_selected.empty:
        raise ValueError("No se encontraron tanques para consultar.")

    tags = df_selected["nivel_tanque"].astype("string").str.strip()
    df_selected["nivel_tanque"] = tags

    df_full_tanks = fetch_mysql_table(
        config=hyd_db_config,
        table="TANQUES",
        columns=["ID_tanque", "Sector", "Distrito", "Tipo", "Vol_util"],
    )

    tanks = df_selected[["TQ", "Nombre_tq", "Nivel_max"]].merge(
        df_full_tanks,
        left_on="TQ",
        right_on="ID_tanque",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    tanks = tanks.rename(
        columns={
            "TQ": "tank_id",
            "Nombre_tq": "name",
            "Nivel_max": "max_level_m",
            "Vol_util": "capacity_m3",
        }
    ).drop(columns=["ID_tanque", "_merge"])

    tanks["max_level_m"] = pd.to_numeric(tanks["max_level_m"], errors="coerce")
    tanks["capacity_m3"] = pd.to_numeric(tanks["capacity_m3"], errors="coerce")

    tank_tags = df_selected.set_index("TQ")["nivel_tanque"].to_dict()
    tank_readings = fetch_tank_readings(
        config=scada_config,
        scada_tags=tank_tags,
        start_date=start_date,
        end_date=end_date,
    )

    return tank_readings, tanks


# =====================================================================
# ESTILOS CSS PERSONALIZADOS
# =====================================================================

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: #081421; color: #e4edf7; }
header[data-testid="stHeader"] { background: #081421; }
[data-testid="stSidebar"] { background: #0c1c2c; border-right: 1px solid #203348; }
.block-container { padding: 3.5rem 2rem 1.4rem; max-width: 1920px; }

h1 { font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.045rem; padding-top: .2rem !important; color: #e4edf7; }
h3 { font-size: 1.03rem !important; font-weight: 600 !important; padding: .1rem 0 !important; color: #e4edf7; }

[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #22384d !important; border-radius: 10px !important; }
[data-testid="stCaptionContainer"] { color: #91a8bf; font-size: .78rem; }

.dma-eyebrow { color: #8fa9c2; letter-spacing: .15em; font-size: .67rem; font-weight: 600; text-transform: uppercase; }
.dma-tab { color: #63b9df; border-bottom: 2px solid #4caacd; width: 95px; padding: 0 0 .5rem .15rem; font-size: .88rem; margin-bottom: .4rem; }

/* Tarjetas nativas KPI */
[class*="st-key-dma_kpi_"] {
    background: linear-gradient(135deg, #122c42 0%, #0b1c2d 100%);
    border: 1px solid #27445e !important;
    border-top: 2px solid #47afd0 !important;
    border-radius: 10px;
    gap: .3rem;
    min-height: 132px;
}
[class*="st-key-dma_kpi_"] [data-testid="stMarkdownContainer"] p { font-size: .85rem; color: #c4d8e9; margin: 0; }
[class*="st-key-dma_kpi_"] [data-testid="stCaptionContainer"] p { font-size: .69rem; color: #91a8bf; margin: 0; }

.dma-kpi-value { color: #f0f7ff; font-size: clamp(1.3rem, 1.65vw, 1.95rem); font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; margin: 2px 0 4px; }
.dma-kpi-value span, .dma-critical-value span { color: #9eb7cd; font-size: .75rem; font-weight: 400; }
.dma-muted { color: #9ab0c7; font-size: .69rem; line-height: 1.55; }

.dma-critical {
    border: 1px solid #244356;
    border-left: 3px solid var(--signal-color, #47afd0);
    background: linear-gradient(120deg, #102b3d, #0c1c2d);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: .8rem;
}
.dma-critical-title { color: #d6e6f4; font-weight: 600; display: flex; gap: 8px; align-items: center; }
.dma-signal-dot { background: var(--signal-color, #47afd0); width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dma-critical-value { font-size: 1.8rem; margin: 4px 0; font-weight: 650; color: var(--signal-color, #47afd0); font-variant-numeric: tabular-nums; }

.dma-condition { display: inline-block; padding: 3px 8px; margin-bottom: 7px; border-radius: 5px; font-size: .7rem; }
.condition-ok { color: #73e3c0; background: #143d36; }
.condition-warn { color: #ffd18a; background: #493923; }
.condition-neutral { color: #b1c3d7; background: #203244; }

.st-key-dma_operational table { width: 100%; border-collapse: collapse; }
.st-key-dma_operational td { border-bottom: 1px solid #23374a; padding: .52rem .35rem; vertical-align: middle; font-size: .8rem; }
.st-key-dma_operational td:first-child { width: 62%; color: #c1d3e5; }
.st-key-dma_operational td:last-child { color: #eef6ff; font-variant-numeric: tabular-nums; }
.st-key-dma_operational td p { font-size: .8rem; margin: 0; }

iframe { border-radius: 8px; }

.tm-card-title { color: #edf3fb; font-size: 1.05rem; font-weight: 600; overflow-wrap: anywhere; }
.tm-card-title b { margin-right: 8px; color: #63b9df; }
.tm-card-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); align-items: start; gap: 8px; margin: 10px 0 4px; }
.tm-card-metrics .tm-metric { display: grid; grid-template-rows: auto auto; min-width: 0; }
.tm-card-metrics .tm-metric span { color: #8fa9c2; font-size: .75rem; line-height: 1.2; }
.tm-card-metrics .tm-metric strong { color: #f0f7ff; font-size: .88rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 3px; }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    [class*="st-key-dma_kpi_"] { min-height: 112px; }
}
</style>
"""

LEVEL_CHART_STYLE = {
    "template": "plotly_dark",
    "height": 220,
    "line_color": "#63b9df",
    "fill_color": "rgba(99, 185, 223, 0.2)",
    "max_color": "#ffd18a",
    "text_color": "#e4edf7",
    "grid_color": "rgba(35, 55, 74, 0.6)",
    "line_width": 1.6,
    "font_size": 11,
}


def _number(value: float, unit: str = "", decimals: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{unit}"


def _prepare_data(readings: pd.DataFrame, tanks: pd.DataFrame, timezone: str):
    data = readings.copy()
    catalog = tanks.copy()

    catalog["tank_id"] = catalog["tank_id"].astype(str).str.strip()
    data["tank_id"] = data["tank_id"].astype(str).str.strip()

    for col in ["capacity_m3", "max_level_m"]:
        catalog[col] = pd.to_numeric(catalog[col], errors="coerce")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["level_m"] = pd.to_numeric(data["level_m"], errors="coerce")

    data = (
        data.drop_duplicates(["tank_id", "timestamp"], keep="last")
        .sort_values(["tank_id", "timestamp"])
        .reset_index(drop=True)
    )
    return data, catalog.set_index("tank_id")


def _render_filters(
    data: pd.DataFrame,
    key: str,
    catalog: pd.DataFrame,
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp | None = None,
):
    first = available_start if available_start is not None else data["timestamp"].min()
    last = available_end if available_end is not None else data["timestamp"].max()

    if pd.isna(first) or pd.isna(last):
        first = pd.Timestamp.now() - pd.Timedelta(days=7)
        last = pd.Timestamp.now()

    quick_key = f"{key}_quick"
    dates_key = f"{key}_dates"
    periods = {"24 h": 1, "7 días": 7, "30 días": 30}

    state = st.session_state
    state.setdefault(quick_key, "7 días")

    def use_custom_dates():
        state[quick_key] = None

    quick = state[quick_key]
    if quick in periods:
        start = max(first, last - pd.Timedelta(days=periods[quick]))
        state[dates_key] = (start.date(), last.date())
    elif dates_key not in state:
        state[dates_key] = (first.date(), last.date())

    categories = sorted(catalog[CATEGORY_COLUMN].dropna().unique().tolist())
    if not categories:
        categories = ["GENERAL"]
        catalog[CATEGORY_COLUMN] = "GENERAL"

    category_key = f"{key}_category"
    if state.get(category_key) not in categories:
        state[category_key] = categories[0]

    with st.container(border=True):
        date_column, quick_column, category_column = st.columns(
            [2, 2, 4], gap="medium", vertical_alignment="bottom"
        )
        with date_column:
            st.date_input(
                "Periodo",
                min_value=first.date(),
                max_value=last.date(),
                format="DD/MM/YYYY",
                key=dates_key,
                on_change=use_custom_dates,
            )
        with quick_column:
            st.segmented_control(
                "Acceso rápido",
                options=list(periods),
                key=quick_key,
                width="stretch",
            )
        with category_column:
            st.segmented_control(
                "Distrito",
                options=categories,
                selection_mode="single",
                required=True,
                key=category_key,
                width="stretch",
            )

    dates = state[dates_key]
    if len(dates) != 2:
        return None

    start = pd.Timestamp(dates[0])
    end = min(
        pd.Timestamp(dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
        last,
    )
    return start, end, state[category_key]


def _summarize(
    data: pd.DataFrame,
    catalog: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_age_minutes: float,
    expected_interval_minutes: float = 5,
    min_communication_pct: float = 80,
):
    interval = pd.Timedelta(minutes=expected_interval_minutes)
    expected_readings = int((end - start) // interval)
    if expected_readings < 1:
        expected_readings = 1

    valid = data.loc[data["level_m"].ge(0)]
    latest = valid.groupby("tank_id", sort=False).tail(1).set_index("tank_id")
    statistics = data.groupby("tank_id")["level_m"].agg(
        min_period="min", max_period="max"
    )

    summary = catalog.join(latest[["timestamp", "level_m"]]).join(statistics)
    usable_geometry = summary["capacity_m3"].gt(0) & summary["max_level_m"].gt(0)

    capacity = summary["capacity_m3"].where(usable_geometry)
    max_level = summary["max_level_m"].where(usable_geometry)
    area_m2 = capacity / max_level

    summary["volume_m3"] = area_m2 * summary["level_m"]
    age = end - summary["timestamp"]
    recent = age.between(pd.Timedelta(0), pd.Timedelta(minutes=max_age_minutes))

    summary["volume_m3"] = summary["volume_m3"].where(recent)
    summary["storage_pct"] = 100 * summary["volume_m3"] / capacity

    coverage_end = start + expected_readings * interval
    received = data.loc[
        data["timestamp"].ge(start)
        & data["timestamp"].lt(coverage_end)
        & data["level_m"].notna(),
        ["tank_id", "timestamp"],
    ].copy()

    if not received.empty:
        received["interval_id"] = (received["timestamp"] - start) // interval
        received_counts = received.groupby("tank_id")["interval_id"].nunique()
    else:
        received_counts = pd.Series(dtype=int)

    summary["expected_readings"] = expected_readings
    summary["received_readings"] = received_counts.reindex(
        summary.index, fill_value=0
    ).astype(int)
    summary["communication_pct"] = (
        100 * summary["received_readings"] / summary["expected_readings"]
    )
    summary["communication_ok"] = summary["communication_pct"].ge(
        min_communication_pct
    )

    return summary


def _render_summary(summary: pd.DataFrame, end: pd.Timestamp):
    available = summary.loc[summary["volume_m3"].notna() & summary["capacity_m3"].gt(0)]
    capacity = available["capacity_m3"].sum(min_count=1)
    volume = available["volume_m3"].sum(min_count=1)
    percent = 100 * volume / capacity if capacity > 0 else np.nan

    insufficient_count = int((~summary["communication_ok"]).sum())
    communicating = summary.loc[summary["communication_ok"]]
    expected_total = communicating["expected_readings"].sum()
    communication_pct = (
        100 * communicating["received_readings"].sum() / expected_total
        if expected_total > 0
        else np.nan
    )

    kpis = [
        ("TANQUES TOTALES", str(len(summary)), "Registrados", "dma_kpi_1"),
        ("COMUNICACIÓN BLA<5%", str(insufficient_count), "Sin reporte activo", "dma_kpi_2"),
        ("% COMUNICACIÓN", _number(communication_pct, " %", 1), "Promedio global", "dma_kpi_3"),
        ("CAPACIDAD REGULACIÓN", _number(capacity, " m³"), "Capacidad instalada", "dma_kpi_4"),
        ("VOLUMEN REGULACIÓN", _number(volume, " m³"), "Volumen actual", "dma_kpi_5"),
        ("% ALMACENAMIENTO", _number(percent, " %", 1), "Nivel de reserva", "dma_kpi_6"),
    ]

    cols = st.columns(6)
    for col, (title, value, caption, key_name) in zip(cols, kpis):
        with col:
            with st.container(border=True, key=key_name):
                st.html(f'<div class="dma-eyebrow">{escape(title)}</div>')
                st.html(f'<div class="dma-kpi-value">{escape(value)}</div>')
                st.caption(caption)

    st.caption(f"Corte de información: {end:%d/%m/%Y %H:%M}")


def _build_level_chart(
    level_data: pd.DataFrame,
    tank_info: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    gap_minutes: float = 60,
):
    settings = LEVEL_CHART_STYLE
    level_data = level_data.sort_values("timestamp").copy()
    fig = go.Figure()

    breaks = (
        level_data["timestamp"].diff().gt(pd.Timedelta(minutes=gap_minutes))
        | level_data["level_m"].isna()
        | level_data["level_m"].shift().isna()
    )
    level_data["_segment"] = breaks.cumsum()
    valid = level_data.dropna(subset=["level_m"])

    for _, block in valid.groupby("_segment"):
        multiple_points = len(block) > 1
        fig.add_trace(
            go.Scatter(
                x=block["timestamp"],
                y=block["level_m"],
                mode="lines+markers" if multiple_points else "markers",
                line={"color": settings["line_color"], "width": settings["line_width"]},
                marker={
                    "color": settings["line_color"],
                    "size": 4,
                    "symbol": "circle",
                    "opacity": 0.8,
                },
                fill="tozeroy" if multiple_points else None,
                fillcolor=settings["fill_color"],
                name="Nivel",
                hovertemplate="<b>%{y:.2f} m</b><br>%{x|%d/%m/%Y %H:%M}<extra></extra>",
            )
        )

    fig.add_hline(
        y=float(tank_info["max_level_m"]),
        line_color=settings["max_color"],
        line_dash="dash",
        line_width=1.2,
    )

    fig.update_layout(
        yaxis_title="m",
        template=settings["template"],
        height=settings["height"],
        margin={"l": 30, "r": 10, "t": 8, "b": 25},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": settings["text_color"], "size": settings["font_size"]},
        showlegend=False,
        hovermode="closest",
    )
    fig.update_xaxes(
        range=[start, end],
        gridcolor=settings["grid_color"],
        zerolinecolor=settings["grid_color"],
    )
    fig.update_yaxes(
        gridcolor=settings["grid_color"],
        zerolinecolor=settings["grid_color"],
    )
    return fig


def _render_tank(
    data: pd.DataFrame,
    tank: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    gap_minutes: float,
    key: str,
):
    display_name = str(tank["name"]).strip()
    tank_id = str(tank.name)

    if display_name.startswith(tank_id):
        display_name = display_name[len(tank_id) :].lstrip(" -")

    st.html(f'<div class="tm-card-title"><b>{escape(tank_id)}</b> {escape(display_name)}</div>')

    metrics_html = f"""
    <div class="tm-card-metrics">
        <div class="tm-metric"><span>Máx. per.</span><strong>{escape(_number(tank['max_period'], ' m', 2))}</strong></div>
        <div class="tm-metric"><span>Mín. per.</span><strong>{escape(_number(tank['min_period'], ' m', 2))}</strong></div>
        <div class="tm-metric"><span>% Com.</span><strong>{escape(_number(tank['communication_pct'], ' %', 1))}</strong></div>
        <div class="tm-metric"><span>% Alm.</span><strong>{escape(_number(tank['storage_pct'], ' %', 0))}</strong></div>
    </div>
    """
    st.html(metrics_html)

    if data["level_m"].notna().any():
        figure = _build_level_chart(data, tank, start, end, gap_minutes)
        st.plotly_chart(
            figure,
            use_container_width=True,
            theme=None,
            key=key,
            config={"displaylogo": False, "displayModeBar": "hover"},
        )
    else:
        st.info("Sin niveles disponibles.")

    timestamp = tank["timestamp"]
    if pd.notna(timestamp):
        elapsed_minutes = max(0, int((end - timestamp).total_seconds() / 60 + 0.5))
        days, rem = divmod(elapsed_minutes, 1440)
        hours, minutes = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes or not parts:
            parts.append(f"{minutes}m")
        elapsed_text = " ".join(parts)
    else:
        elapsed_text = "Sin datos"

    st.caption(f"Antigüedad de lectura: {elapsed_text}")


def render_monitoring(
    readings: pd.DataFrame,
    tanks: pd.DataFrame,
    *,
    columns: int = 3,
    gap_minutes: float = 90,
    timezone: str = "America/Mexico_City",
    key: str = "tank_monitoring",
    available_start: pd.Timestamp | None = None,
    available_end: pd.Timestamp | None = None,
):
    try:
        data, catalog = _prepare_data(readings, tanks, timezone)
        if CATEGORY_COLUMN not in catalog.columns:
            catalog[CATEGORY_COLUMN] = "GENERAL"

        st.html(CUSTOM_CSS)

        period = _render_filters(
            data, key, catalog, available_start, available_end
        )
        if period is None:
            return None

        start, end, category = period
        catalog = catalog.loc[catalog[CATEGORY_COLUMN].eq(category)].copy()
        filtered = data.loc[
            data["timestamp"].between(start, end)
            & data["tank_id"].isin(catalog.index)
        ].copy()

        summary = _summarize(
            data=filtered,
            catalog=catalog,
            start=start,
            end=end,
            max_age_minutes=MAX_AGE_MINUTES,
        )
        _render_summary(summary, end)

        st.markdown("### Detalle por tanque")

        tank_ids = catalog.index.tolist()
        for offset in range(0, len(tank_ids), columns):
            row_columns = st.columns(columns)
            for column, tank_id in zip(
                row_columns, tank_ids[offset : offset + columns]
            ):
                with column, st.container(border=True):
                    try:
                        _render_tank(
                            data=filtered.loc[filtered["tank_id"].eq(tank_id)],
                            tank=summary.loc[tank_id],
                            start=start,
                            end=end,
                            gap_minutes=gap_minutes,
                            key=f"{key}_chart_{tank_id}",
                        )
                    except Exception:
                        LOGGER.exception("Error al renderizar tanque %s", tank_id)
                        st.error(f"No fue posible mostrar {tank_id}.")
        return summary
    except Exception as exc:
        LOGGER.exception("Error en render_monitoring")
        st.error(f"Error al procesar el tablero: {exc}")
    return None


# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================


def main():
    st.set_page_config(
        page_title="Monitoreo de Tanques - MIAA",
        page_icon=":material/account_tree:",
        layout="wide",
    )

    st.title("Monitoreo de Tanques de Regulación - MIAA")

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    cutoff = now.floor("5min")
    window_start = cutoff - pd.Timedelta(days=DEFAULT_LOAD_DAYS)

    try:
        with st.spinner("Cargando lecturas y catálogo de tanques..."):
            readings, tanks = get_tank_level_data(
                start_date=window_start, end_date=cutoff, ntanks=None
            )

        render_monitoring(
            readings=readings,
            tanks=tanks,
            timezone=TIMEZONE,
            available_start=window_start,
            available_end=cutoff,
        )
    except Exception as e:
        LOGGER.exception("Error crítico al cargar los datos.")
        st.error(f"Error al conectar o consultar la base de datos: {e}")

    st.divider()
    st.caption(
        "**Dirección de Planeación** | Subdirección de Tecnología e Innovación"
    )


if __name__ == "__main__":
    main()
