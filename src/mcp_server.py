"""Servidor MCP para Mantenimiento Predictivo e Ignition Perspective HMI sobre NASA C-MAPSS.

Expone herramientas (tools) y recursos (resources) del Gemelo Digital para:
1. Predicción de RUL y diagnóstico de deriva física de sensores sobre C-MAPSS.
2. Auditoría, diseño Dark-Mode ISA-101 con Dibujo Vectorial Animado de la Turbina (SVG con
   rotores girando, flujos de aire, llama pulsante y desgaste térmico por colores),
   Simulador Interactivo de Ciclos de Vuelo (Slider 1..192) y despliegue en Ignition Perspective.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "cmapss"
MODELS_DIR = BASE_DIR / "models"
SCADA_DIR = BASE_DIR / "scada" / "perspective"
MANIFEST_PATH = DATA_DIR / "manifest.yaml"

sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from adapter import load
from train_cmapss import nasa_score, train_and_evaluate

_DATA_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_MODEL_CACHE: dict[str, xgb.XGBRegressor] = {}
_MANIFEST_CACHE: dict[str, Any] | None = None


def _get_manifest() -> dict[str, Any]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            _MANIFEST_CACHE = yaml.safe_load(f)
    return _MANIFEST_CACHE


def _get_dataset(subset: str = "FD001", split: str = "test") -> pd.DataFrame:
    subset = subset.upper()
    split = split.lower()
    key = (subset, split)
    if key not in _DATA_CACHE:
        manifest = _get_manifest()
        rul_cap = int(manifest.get("rul_cap", 125))
        with redirect_stdout(sys.stderr):
            df = load(str(DATA_DIR), rul_cap=rul_cap, subset=subset, split=split)
            df = df.copy()
            df["cycle"] = df.groupby("asset_id").cumcount() + 1
        _DATA_CACHE[key] = df
    return _DATA_CACHE[key]


def _get_model(subset: str = "FD001") -> xgb.XGBRegressor:
    subset = subset.upper()
    if subset not in _MODEL_CACHE:
        model_path = MODELS_DIR / f"xgb_cmapss_{subset.lower()}.json"
        if model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
        else:
            with redirect_stdout(sys.stderr):
                model, _ = train_and_evaluate(subset=subset, save_model=True)
        _MODEL_CACHE[subset] = model
    return _MODEL_CACHE[subset]


def _classify_health(rul: float) -> tuple[str, str]:
    if rul <= 25.0:
        return (
            "CRITICAL",
            "Programar parada inmediata e inspección boroscópica de turbina LPT/HPC.",
        )
    if rul <= 60.0:
        return (
            "WARNING",
            "Incrementar frecuencia de monitoreo térmico y planificar mantenimiento en próxima ventana.",
        )
    return (
        "HEALTHY",
        "Operación nominal. Continuar monitoreo predictivo estándar.",
    )


def get_fleet_status(subset: str = "FD001", split: str = "test", top_n: int = 10) -> dict[str, Any]:
    """Devuelve el resumen de salud de toda la flota de turbinas y las unidades más críticas."""
    df = _get_dataset(subset=subset, split=split)
    model = _get_model(subset=subset)

    last_df = df.groupby("asset_id", sort=False).tail(1).copy()
    X_last = pd.DataFrame(last_df["features"].tolist(), index=last_df.index)
    preds = model.predict(X_last)

    last_df["predicted_rul"] = np.clip(preds, 0.0, None)
    last_df["error_cycles"] = last_df["predicted_rul"] - last_df["target"]

    records = []
    counts = {"CRITICAL": 0, "WARNING": 0, "HEALTHY": 0}
    for _, row in last_df.iterrows():
        pred_rul = float(row["predicted_rul"])
        status, action = _classify_health(pred_rul)
        counts[status] += 1
        records.append(
            {
                "asset_id": str(row["asset_id"]),
                "last_cycle": int(row["cycle"]),
                "predicted_rul": round(pred_rul, 2),
                "true_rul": round(float(row["target"]), 2),
                "error_cycles": round(float(row["error_cycles"]), 2),
                "status": status,
                "recommended_action": action,
            }
        )

    records_sorted = sorted(records, key=lambda r: r["predicted_rul"])
    return {
        "dataset": f"cmapss_{subset.lower()}",
        "split": split,
        "total_engines": len(records),
        "status_summary": counts,
        "fleet_mean_predicted_rul": round(float(np.mean([r["predicted_rul"] for r in records])), 2),
        "most_critical_engines": records_sorted[: int(top_n)],
    }


def predict_engine_rul(
    asset_id: str,
    cycle: int | None = None,
    subset: str = "FD001",
    split: str = "test",
) -> dict[str, Any]:
    """Predice el RUL (Remaining Useful Life) para un motor específico y explica los sensores clave."""
    asset_id_str = str(asset_id).replace("cmapss_", "").strip()
    df = _get_dataset(subset=subset, split=split)
    engine_df = df[df["asset_id"] == asset_id_str]

    if engine_df.empty:
        available = sorted(df["asset_id"].unique().tolist(), key=lambda x: int(x))
        raise ValueError(
            f"Motor '{asset_id}' no encontrado en {subset} ({split}). "
            f"Rango disponible: {available[0]}..{available[-1]}"
        )

    max_cycle = int(engine_df["cycle"].max())
    if cycle is None:
        target_row = engine_df.iloc[-1]
    else:
        matched = engine_df[engine_df["cycle"] == int(cycle)]
        if matched.empty:
            raise ValueError(
                f"El ciclo {cycle} no existe para el motor {asset_id_str} (ciclos disponibles: 1..{max_cycle})."
            )
        target_row = matched.iloc[0]

    model = _get_model(subset=subset)
    features_dict = target_row["features"]
    X_single = pd.DataFrame([features_dict])
    pred_rul = float(np.clip(model.predict(X_single)[0], 0.0, None))
    true_rul = float(target_row["target"])
    status, action = _classify_health(pred_rul)

    manifest = _get_manifest()
    descriptions = manifest.get("descriptions", {})
    importances = dict(zip(X_single.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

    key_sensors = [
        {
            "sensor": feat_name,
            "description": descriptions.get(feat_name, feat_name),
            "value": round(float(features_dict[feat_name]), 4),
            "model_importance": round(float(imp), 4),
        }
        for feat_name, imp in top_features
    ]

    return {
        "asset_id": asset_id_str,
        "dataset": f"cmapss_{subset.lower()}",
        "split": split,
        "evaluated_cycle": int(target_row["cycle"]),
        "max_recorded_cycle": max_cycle,
        "predicted_rul_cycles": round(pred_rul, 2),
        "true_rul_cycles": round(true_rul, 2),
        "prediction_error_cycles": round(pred_rul - true_rul, 2),
        "health_status": status,
        "recommended_action": action,
        "top_influential_sensors": key_sensors,
    }


def get_sensor_telemetry(
    asset_id: str,
    last_n_cycles: int = 10,
    subset: str = "FD001",
    split: str = "test",
) -> dict[str, Any]:
    """Obtiene el historial reciente de telemetría de un motor y detecta qué sensores presentan mayor deriva física."""
    asset_id_str = str(asset_id).replace("cmapss_", "").strip()
    df = _get_dataset(subset=subset, split=split)
    engine_df = df[df["asset_id"] == asset_id_str]

    if engine_df.empty:
        raise ValueError(f"Motor '{asset_id}' no encontrado en {subset} ({split}).")

    features_df = pd.DataFrame(engine_df["features"].tolist())
    features_df["cycle"] = engine_df["cycle"].to_numpy()

    manifest = _get_manifest()
    descriptions = manifest.get("descriptions", {})
    sensor_cols = [f"s{i}" for i in range(1, 22)]

    baseline_window = min(10, len(features_df))
    recent_window = min(int(last_n_cycles), len(features_df))
    baseline = features_df[sensor_cols].iloc[:baseline_window]
    recent = features_df[sensor_cols].iloc[-recent_window:]

    drifts = []
    for s_col in sensor_cols:
        base_mean = float(baseline[s_col].mean())
        base_std = float(baseline[s_col].std(ddof=0))
        rec_mean = float(recent[s_col].mean())
        last_val = float(recent[s_col].iloc[-1])
        delta = rec_mean - base_mean

        if abs(delta) < 1e-9 and base_std < 1e-9:
            continue

        z_drift = delta / base_std if base_std > 1e-6 else 0.0
        pct_change = (delta / base_mean) * 100.0 if abs(base_mean) > 1e-9 else 0.0
        drifts.append(
            {
                "sensor": s_col,
                "description": descriptions.get(s_col, s_col),
                "baseline_mean": round(base_mean, 4),
                "recent_mean": round(rec_mean, 4),
                "latest_reading": round(last_val, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct_change, 3),
                "z_score_drift": round(z_drift, 2),
                "recent_series": [round(float(v), 4) for v in recent[s_col].tolist()],
            }
        )

    drifts_sorted = sorted(drifts, key=lambda d: abs(d["z_score_drift"]), reverse=True)
    return {
        "asset_id": asset_id_str,
        "dataset": f"cmapss_{subset.lower()}",
        "split": split,
        "total_recorded_cycles": int(len(features_df)),
        "window_cycles": [int(c) for c in features_df["cycle"].iloc[-recent_window:].tolist()],
        "top_degrading_sensors": drifts_sorted[:8],
    }


def get_model_diagnostics(subset: str = "FD001") -> dict[str, Any]:
    """Devuelve las métricas globales del modelo RUL (RMSE, MAE, R², NASA S-Score) y el ranking de importancia de variables."""
    model = _get_model(subset=subset)
    test_df = _get_dataset(subset=subset, split="test")

    X_test = pd.DataFrame(test_df["features"].tolist(), index=test_df.index)
    y_test = test_df["target"].to_numpy()
    preds_all = np.clip(model.predict(X_test), 0.0, None)

    last_idx = test_df.groupby("asset_id", sort=False).tail(1).index
    y_last = test_df.loc[last_idx, "target"].to_numpy()
    preds_last = preds_all[test_df.index.get_indexer(last_idx)]

    rmse_last = float(np.sqrt(np.mean((preds_last - y_last) ** 2)))
    mae_last = float(np.mean(np.abs(preds_last - y_last)))
    ss_res = float(np.sum((y_last - preds_last) ** 2))
    ss_tot = float(np.sum((y_last - np.mean(y_last)) ** 2))
    r2_last = 1.0 - (ss_res / ss_tot)
    s_score = nasa_score(y_last, preds_last)

    manifest = _get_manifest()
    descriptions = manifest.get("descriptions", {})
    importances = sorted(
        zip(X_test.columns, model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {
        "model_type": "XGBRegressor",
        "subset": subset.upper(),
        "test_engines_count": int(len(y_last)),
        "metrics_last_cycle": {
            "rmse_cycles": round(rmse_last, 2),
            "mae_cycles": round(mae_last, 2),
            "r2": round(r2_last, 4),
            "nasa_s_score": round(s_score, 2),
        },
        "top_10_features": [
            {
                "feature": feat,
                "description": descriptions.get(feat, feat),
                "importance": round(float(imp), 4),
            }
            for feat, imp in importances[:10]
        ],
    }


def audit_perspective_hmi(
    container_name: str = "twin-ignition",
    project: str = "DigitalTwin",
    view_name: str = "Turbofan_Overview",
) -> dict[str, Any]:
    """Inspecciona la vista actual de Ignition Perspective en el contenedor Docker y detecta bugs de bindings y mejoras ISA-101."""
    view_path = f"/usr/local/bin/ignition/data/projects/{project}/com.inductiveautomation.perspective/views/{view_name}/view.json"
    cmd = ["docker", "exec", container_name, "cat", view_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"No se pudo leer {view_path} en {container_name}: {proc.stderr.strip()}")

    raw_json = proc.stdout or ""
    issues = []

    if "ia.display.simple-gauge" in raw_json:
        issues.append(
            {
                "severity": "COMPONENT_TYPE_ERROR",
                "component": "Gauge_RUL",
                "issue": "Tipo 'ia.display.simple-gauge' inválido en Perspective; debe ser 'ia.chart.simple-gauge'.",
            }
        )

    if "../" in raw_json or ".../" in raw_json or "..../" in raw_json:
        issues.append(
            {
                "severity": "FRAGILE_RELATIVE_BINDING",
                "component": "View Bindings",
                "issue": "Uso de rutas relativas ('../', '.../') propensas a PropertyReference failed; usar 'view.custom.*'.",
            }
        )

    return {
        "container": container_name,
        "project": project,
        "view_name": view_name,
        "issues_found": len(issues),
        "issues": issues,
    }


def _build_dark_kpi_card(
    name: str,
    title: str,
    subtitle: str,
    value_binding_path: str | None = None,
    value_expression: str | None = None,
    accent_color: str = "#38BDF8",
    is_health_card: bool = False,
) -> dict[str, Any]:
    """Construye una tarjeta KPI Dark-Mode Aeroespacial usando rutas absolutas view.custom."""
    val_prop_config: dict[str, Any] = {}
    if value_binding_path:
        val_prop_config["props.text"] = {
            "binding": {
                "type": "property",
                "config": {"path": value_binding_path},
            }
        }
    elif value_expression:
        val_prop_config["props.text"] = {
            "binding": {
                "type": "expr",
                "config": {"expression": value_expression},
            }
        }

    if is_health_card:
        val_prop_config["props.style.color"] = {
            "binding": {
                "type": "expr",
                "config": {
                    "expression": (
                        "if({view.custom.kpis[0].health_status} = 'Critico', '#F87171', "
                        "if({view.custom.kpis[0].health_status} = 'Alerta', '#FBBF24', '#4ADE80'))"
                    )
                },
            }
        }

    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": "195px", "grow": 1},
        "props": {
            "direction": "column",
            "justify": "space-between",
            "style": {
                "backgroundColor": "#1E293B",
                "borderRadius": "10px",
                "border": "1px solid #334155",
                "borderTop": f"4px solid {accent_color}",
                "padding": "8px 14px",
                "margin": "3px",
                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
            },
        },
        "children": [
            {
                "meta": {"name": "Lbl_Titulo"},
                "type": "ia.display.label",
                "position": {"basis": "16px"},
                "props": {
                    "text": title,
                    "style": {
                        "color": "#94A3B8",
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "letterSpacing": "0.06em",
                        "textTransform": "uppercase",
                    },
                },
            },
            {
                "meta": {"name": "Lbl_Valor"},
                "type": "ia.display.label",
                "position": {"basis": "32px", "grow": 1},
                "propConfig": val_prop_config,
                "props": {
                    "style": {
                        "color": "#F8FAFC",
                        "fontSize": "23px",
                        "fontWeight": "800",
                        "letterSpacing": "-0.01em",
                    }
                },
            },
            {
                "meta": {"name": "Lbl_Subtitulo"},
                "type": "ia.display.label",
                "position": {"basis": "15px"},
                "props": {
                    "text": subtitle,
                    "style": {
                        "color": "#64748B",
                        "fontSize": "10px",
                        "fontWeight": "500",
                    },
                },
            },
        ],
    }


def _build_animated_turbofan_svg_expr() -> str:
    """Construye la expresión de Ignition que genera el SVG CAD 2D Avanzado (Opción A) con gradientes metálicos 3D, partículas y pines HUD."""
    parts = [
        'concat("data:image/svg+xml;utf8,',
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 920 260' width='100%' height='100%' style='background:rgb(9,14,26);border-radius:8px;font-family:Segoe UI,sans-serif'>",
        # Definición de Gradientes Metálicos Cilíndricos 3D (usando %23 para '#' en URL data URI)
        "<defs>",
        "<linearGradient id='metalNacelle' x1='0%' y1='0%' x2='0%' y2='100%'>",
        "<stop offset='0%' stop-color='rgb(148,163,184)'/><stop offset='45%' stop-color='rgb(71,85,105)'/><stop offset='100%' stop-color='rgb(30,41,59)'/>",
        "</linearGradient>",
        "<linearGradient id='chromeShaft' x1='0%' y1='0%' x2='0%' y2='100%'>",
        "<stop offset='0%' stop-color='rgb(226,232,240)'/><stop offset='50%' stop-color='rgb(100,116,139)'/><stop offset='100%' stop-color='rgb(203,213,225)'/>",
        "</linearGradient>",
        "<path id='flowTop' d='M 20,76 Q 260,54 450,78 T 880,82' fill='none'/>",
        "<path id='flowBot' d='M 20,184 Q 260,206 450,182 T 880,178' fill='none'/>",
        "</defs>",
        # Retícula Blueprint CAD Aeroespacial
        "<line x1='0' y1='130' x2='920' y2='130' stroke='rgb(30,41,59)' stroke-width='1.5' stroke-dasharray='8,6'/>",
        "<line x1='115' y1='25' x2='115' y2='235' stroke='rgb(30,41,59)' stroke-width='1' stroke-dasharray='4,4'/>",
        "<line x1='340' y1='25' x2='340' y2='235' stroke='rgb(30,41,59)' stroke-width='1' stroke-dasharray='4,4'/>",
        "<line x1='615' y1='25' x2='615' y2='235' stroke='rgb(30,41,59)' stroke-width='1' stroke-dasharray='4,4'/>",
        # Etiqueta Superior Opción A
        "<text x='14' y='20' fill='rgb(56,189,248)' font-size='11' font-weight='bold' letter-spacing='0.8'>● OPCIÓN A: PLANO CAD VECTORIAL 2D + PARTÍCULAS + PINES HUD</text>",
        # Carcasa Metálica 3D (Nacelle Superior e Inferior)
        "<path d='M 45,44 Q 210,24 430,36 L 690,56 L 785,76 L 785,90 L 685,70 L 430,52 Q 210,40 55,58 Z' fill='url(%23metalNacelle)' stroke='rgb(148,163,184)' stroke-width='1.5'/>",
        "<path d='M 45,216 Q 210,236 430,224 L 690,204 L 785,184 L 785,170 L 685,190 L 430,208 Q 210,220 55,202 Z' fill='url(%23metalNacelle)' stroke='rgb(148,163,184)' stroke-width='1.5'/>",
        # Pluma de Gases de Escape Dinámica (Exhaust Plume)
        "<polygon points='770,84 910,60 910,200 770,176' fill='\", ",
        'if({view.custom.kpis[0].lpt_temp_s4} >= 1418.0, "rgb(239,68,68)", if({view.custom.kpis[0].lpt_temp_s4} >= 1408.0, "rgb(245,158,11)", "rgb(56,189,248)")), ',
        "\"' opacity='0.26'><animate attributeName='opacity' values='0.15;0.38;0.15' dur='0.5s' repeatCount='indefinite'/></polygon>",
        # Partículas Termodinámicas viajando por el interior del motor (<animateMotion>)
        "<circle r='3.5' fill='rgb(56,189,248)'><animateMotion dur='1.4s' repeatCount='indefinite' path='M 20,76 Q 240,62 420,86'/></circle>",
        "<circle r='3.5' fill='rgb(56,189,248)'><animateMotion dur='1.4s' begin='0.45s' repeatCount='indefinite' path='M 20,184 Q 240,198 420,174'/></circle>",
        "<circle r='4' fill='rgb(251,191,36)'><animateMotion dur='0.9s' repeatCount='indefinite' path='M 280,92 L 540,98'/></circle>",
        "<circle r='4' fill='rgb(239,68,68)'><animateMotion dur='0.65s' repeatCount='indefinite' path='M 460,96 L 895,78'/></circle>",
        "<circle r='4' fill='rgb(239,68,68)'><animateMotion dur='0.65s' begin='0.3s' repeatCount='indefinite' path='M 460,164 L 895,182'/></circle>",
        # Eje Central Cromado (Core Shaft)
        "<rect x='90' y='121' width='655' height='18' rx='4' fill='url(%23chromeShaft)' stroke='rgb(71,85,105)' stroke-width='1.5'/>",
        "<line x1='100' y1='130' x2='735' y2='130' stroke='rgb(15,23,42)' stroke-width='3.5' stroke-dasharray='16,10'><animate attributeName='stroke-dashoffset' from='52' to='0' dur='0.4s' repeatCount='indefinite'/></line>",
        # Etapa 2: LPC (Compresor Baja Presión)
        "<polygon points='175,68 258,76 258,184 175,192' fill='\", ",
        'if({view.custom.kpis[0].lpc_temp_s2} >= 643.4, "rgb(127,29,29)", if({view.custom.kpis[0].lpc_temp_s2} >= 642.8, "rgb(120,53,15)", "rgb(15,118,110)")), ',
        "\"' stroke='\", ",
        'if({view.custom.kpis[0].lpc_temp_s2} >= 643.4, "rgb(248,113,113)", if({view.custom.kpis[0].lpc_temp_s2} >= 642.8, "rgb(251,191,36)", "rgb(45,212,191)")), ',
        "\"' stroke-width='2' opacity='0.9'/>",
        "<line x1='198' y1='71' x2='198' y2='189' stroke='rgb(226,232,240)' stroke-width='3' stroke-dasharray='6,4'><animate attributeName='stroke-dashoffset' from='20' to='0' dur='0.45s' repeatCount='indefinite'/></line>",
        "<line x1='232' y1='74' x2='232' y2='186' stroke='rgb(226,232,240)' stroke-width='3' stroke-dasharray='6,4'><animate attributeName='stroke-dashoffset' from='0' to='20' dur='0.45s' repeatCount='indefinite'/></line>",
        # Etapa 3: HPC (Compresor Alta Presión ★ 52% IA)
        "<polygon points='272,76 412,84 412,176 272,184' fill='\", ",
        'if({view.custom.kpis[0].hpc_static_press_s11} >= 47.85, "rgb(153,27,27)", if({view.custom.kpis[0].hpc_static_press_s11} >= 47.55, "rgb(146,64,14)", "rgb(6,95,70)")), ',
        "\"' stroke='\", ",
        'if({view.custom.kpis[0].hpc_static_press_s11} >= 47.85, "rgb(239,68,68)", if({view.custom.kpis[0].hpc_static_press_s11} >= 47.55, "rgb(245,158,11)", "rgb(16,185,129)")), ',
        "\"' stroke-width='2.5'/>",
        "<line x1='298' y1='78' x2='298' y2='182' stroke='rgb(248,250,252)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='18' to='0' dur='0.3s' repeatCount='indefinite'/></line>",
        "<line x1='338' y1='80' x2='338' y2='180' stroke='rgb(248,250,252)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='0' to='18' dur='0.3s' repeatCount='indefinite'/></line>",
        "<line x1='378' y1='82' x2='378' y2='178' stroke='rgb(248,250,252)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='18' to='0' dur='0.3s' repeatCount='indefinite'/></line>",
        # Etapa 4: Cámara de Combustión con Llama Anular Pulsante
        "<rect x='426' y='80' width='118' height='100' rx='10' fill='rgb(30,41,59)' stroke='rgb(249,115,22)' stroke-width='2'/>",
        "<ellipse cx='485' cy='100' rx='42' ry='13' fill='rgb(249,115,22)'><animate attributeName='rx' values='36;48;36' dur='0.4s' repeatCount='indefinite'/><animate attributeName='opacity' values='0.65;1;0.65' dur='0.4s' repeatCount='indefinite'/></ellipse>",
        "<ellipse cx='485' cy='160' rx='42' ry='13' fill='rgb(249,115,22)'><animate attributeName='rx' values='36;48;36' dur='0.4s' repeatCount='indefinite'/><animate attributeName='opacity' values='0.65;1;0.65' dur='0.4s' repeatCount='indefinite'/></ellipse>",
        # Etapa 5: Turbina LPT & Escape (★ 10% IA)
        "<polygon points='558,84 692,68 692,192 558,176' fill='\", ",
        'if({view.custom.kpis[0].lpt_temp_s4} >= 1418.0, "rgb(153,27,27)", if({view.custom.kpis[0].lpt_temp_s4} >= 1408.0, "rgb(146,64,14)", "rgb(6,95,70)")), ',
        "\"' stroke='\", ",
        'if({view.custom.kpis[0].lpt_temp_s4} >= 1418.0, "rgb(239,68,68)", if({view.custom.kpis[0].lpt_temp_s4} >= 1408.0, "rgb(245,158,11)", "rgb(16,185,129)")), ',
        "\"' stroke-width='2.5'/>",
        "<line x1='590' y1='82' x2='590' y2='178' stroke='rgb(254,215,170)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='18' to='0' dur='0.3s' repeatCount='indefinite'/></line>",
        "<line x1='632' y1='76' x2='632' y2='184' stroke='rgb(254,215,170)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='0' to='18' dur='0.3s' repeatCount='indefinite'/></line>",
        "<line x1='672' y1='71' x2='672' y2='189' stroke='rgb(254,215,170)' stroke-width='3' stroke-dasharray='5,4'><animate attributeName='stroke-dashoffset' from='18' to='0' dur='0.3s' repeatCount='indefinite'/></line>",
        "<polygon points='692,98 775,130 692,162' fill='url(%23metalNacelle)' stroke='rgb(148,163,184)' stroke-width='1.5'/>",
        # Etapa 1: Rotor del Fan con 8 Aspas Aerodinámicas Girando + Ojiva Cónica
        "<ellipse cx='115' cy='130' rx='26' ry='74' fill='rgb(15,23,42)' stroke='rgb(56,189,248)' stroke-width='2.5'/>",
        "<g transform='translate(115,130)'><g><animateTransform attributeName='transform' type='rotate' from='0' to='360' dur='1.4s' repeatCount='indefinite'/>",
        "<line x1='-20' y1='-62' x2='20' y2='62' stroke='rgb(56,189,248)' stroke-width='5.5' stroke-linecap='round'/>",
        "<line x1='20' y1='-62' x2='-20' y2='62' stroke='rgb(56,189,248)' stroke-width='5.5' stroke-linecap='round'/>",
        "<line x1='0' y1='-68' x2='0' y2='68' stroke='rgb(186,230,253)' stroke-width='4.5' stroke-linecap='round'/>",
        "<line x1='-24' y1='0' x2='24' y2='0' stroke='rgb(186,230,253)' stroke-width='4.5' stroke-linecap='round'/>",
        "</g></g>",
        "<polygon points='48,130 104,108 104,152' fill='url(%23chromeShaft)' stroke='rgb(56,189,248)' stroke-width='1.5'/>",
        # Pines HUD con Líneas de Puntero y Dianas Pulsantes
        "<line x1='340' y1='95' x2='340' y2='28' stroke='rgb(239,68,68)' stroke-width='1.5'/>",
        "<circle cx='340' cy='95' r='5' fill='rgb(239,68,68)'><animate attributeName='r' values='4;9;4' dur='0.9s' repeatCount='indefinite'/></circle>",
        "<rect x='255' y='6' width='185' height='24' rx='4' fill='rgb(15,23,42)' stroke='\", ",
        'if({view.custom.kpis[0].hpc_static_press_s11} >= 47.85, "rgb(239,68,68)", "rgb(16,185,129)"), ',
        "\"' stroke-width='1.5'/><text x='263' y='22' fill='rgb(248,250,252)' font-size='11' font-weight='bold'>★ HPC s11: \", ",
        'numberFormat({view.custom.kpis[0].hpc_static_press_s11}, "0.00"), " psia</text>", ',
        "\"<line x1='625' y1='95' x2='625' y2='28' stroke='rgb(249,115,22)' stroke-width='1.5'/>",
        "<circle cx='625' cy='95' r='5' fill='rgb(249,115,22)'><animate attributeName='r' values='4;9;4' dur='0.9s' repeatCount='indefinite'/></circle>",
        "<rect x='540' y='6' width='180' height='24' rx='4' fill='rgb(15,23,42)' stroke='\", ",
        'if({view.custom.kpis[0].lpt_temp_s4} >= 1418.0, "rgb(239,68,68)", "rgb(16,185,129)"), ',
        "\"' stroke-width='1.5'/><text x='548' y='22' fill='rgb(248,250,252)' font-size='11' font-weight='bold'>★ LPT s4: \", ",
        'numberFormat({view.custom.kpis[0].lpt_temp_s4}, "0.0"), " °R</text>", ',
        "\"<rect x='155' y='230' width='145' height='24' rx='4' fill='rgb(15,23,42)' stroke='rgb(45,212,191)' stroke-width='1'/>",
        "<text x='163' y='246' fill='rgb(248,250,252)' font-size='11' font-weight='bold'>LPC s2: \", ",
        'numberFormat({view.custom.kpis[0].lpc_temp_s2}, "0.0"), " °R</text>", ',
        "\"<rect x='415' y='230' width='165' height='24' rx='4' fill='rgb(15,23,42)' stroke='rgb(249,115,22)' stroke-width='1'/>",
        "<text x='423' y='246' fill='rgb(248,250,252)' font-size='11' font-weight='bold'>CORE s9: \", ",
        'numberFormat({view.custom.kpis[0].core_speed_s9}, "0.0"), " rpm</text>", ',
        "\"<text x='790' y='122' fill='rgb(248,250,252)' font-size='12' font-weight='bold'>CICLO: \", ",
        '{view.custom.kpis[0].current_cycle}, "</text>", ',
        "\"<text x='790' y='144' fill='\", ",
        'if({view.custom.kpis[0].health_status} = "Critico", "rgb(248,113,113)", if({view.custom.kpis[0].health_status} = "Alerta", "rgb(251,191,36)", "rgb(74,222,128)")), ',
        "\"' font-size='12' font-weight='bold'>\", upper({view.custom.kpis[0].health_status}), \"</text></svg>\")",
    ]
    return "".join(parts)


def _build_turbofan_synoptic_section() -> dict[str, Any]:
    """Construye el panel comparativo Lado a Lado: [Opción A: SVG CAD 2D] vs [Opción B: Gemelo 3D WebGL Interactivo]."""
    iframe_src_expr = (
        'concat("http://localhost:8095/turbine3d.html#asset=", {view.custom.selectedAsset}, '
        '"&cycle=", {view.custom.kpis[0].current_cycle}, '
        '"&rul=", numberFormat({view.custom.kpis[0].rul_cycles}, "0.0"), '
        '"&s11=", numberFormat({view.custom.kpis[0].hpc_static_press_s11}, "0.00"), '
        '"&s4=", numberFormat({view.custom.kpis[0].lpt_temp_s4}, "0.0"), '
        '"&s2=", numberFormat({view.custom.kpis[0].lpc_temp_s2}, "0.0"), '
        '"&s8=", numberFormat({view.custom.kpis[0].fan_speed_s8}, "0.0"), '
        '"&s9=", numberFormat({view.custom.kpis[0].core_speed_s9}, "0.0"), '
        '"&status=", {view.custom.kpis[0].health_status})'
    )

    return {
        "meta": {"name": "Turbofan_Synoptic_Container"},
        "type": "ia.container.flex",
        "position": {"basis": "325px", "shrink": 0},
        "props": {
            "direction": "column",
            "style": {
                "backgroundColor": "#1E293B",
                "borderRadius": "10px",
                "border": "1px solid #334155",
                "padding": "10px 16px",
                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                "gap": "8px",
            },
        },
        "children": [
            # Barra superior con el Slider de Simulación de Ciclos sincronizado con ambas vistas (2D y 3D)
            {
                "meta": {"name": "Synoptic_Header_Row"},
                "type": "ia.container.flex",
                "position": {"basis": "32px", "shrink": 0},
                "props": {
                    "direction": "row",
                    "justify": "space-between",
                    "alignItems": "center",
                    "style": {"gap": "14px"},
                },
                "children": [
                    {
                        "meta": {"name": "Lbl_Synoptic_Title"},
                        "type": "ia.display.label",
                        "position": {"basis": "500px", "grow": 1},
                        "props": {
                            "text": "🔬 COMPARATIVA DE GEMELO VISUAL: [OPCIÓN A: PLANO CAD SVG 2D]  VS  [OPCIÓN B: VISOR 3D INTERACTIVO WEBGL]",
                            "style": {
                                "color": "#F8FAFC",
                                "fontSize": "12px",
                                "fontWeight": "800",
                                "letterSpacing": "0.04em",
                            },
                        },
                    },
                    {
                        "meta": {"name": "Lbl_Slider_Info"},
                        "type": "ia.display.label",
                        "position": {"basis": "230px", "shrink": 0},
                        "propConfig": {
                            "props.text": {
                                "binding": {
                                    "type": "expr",
                                    "config": {
                                        "expression": 'concat("⏱️ SIMULAR CICLO DE VUELO: ", {view.custom.simCycle})'
                                    },
                                }
                            }
                        },
                        "props": {
                            "style": {
                                "color": "#38BDF8",
                                "fontSize": "12px",
                                "fontWeight": "800",
                                "textAlign": "right",
                            }
                        },
                    },
                    {
                        "meta": {"name": "Slider_SimCycle"},
                        "type": "ia.input.slider",
                        "position": {"basis": "320px", "shrink": 0},
                        "propConfig": {
                            "props.value": {
                                "binding": {
                                    "type": "property",
                                    "config": {
                                        "bidirectional": True,
                                        "path": "view.custom.simCycle",
                                    },
                                }
                            }
                        },
                        "props": {
                            "min": 1,
                            "max": 192,
                            "step": 1,
                            "color": "#38BDF8",
                        },
                    },
                ],
            },
            # Contenedor Dividido 50% / 50%: Izquierda = SVG CAD 2D | Derecha = Visor 3D WebGL
            {
                "meta": {"name": "Dual_Comparison_Row"},
                "type": "ia.container.flex",
                "position": {"basis": "268px", "grow": 1},
                "props": {
                    "direction": "row",
                    "style": {"gap": "12px"},
                },
                "children": [
                    {
                        "meta": {"name": "OptionA_SVG_CAD"},
                        "type": "ia.display.image",
                        "position": {"basis": "48%", "grow": 1},
                        "propConfig": {
                            "props.source": {
                                "binding": {
                                    "type": "expr",
                                    "config": {
                                        "expression": _build_animated_turbofan_svg_expr(),
                                    },
                                }
                            }
                        },
                        "props": {
                            "fit": {"mode": "contain"},
                            "style": {
                                "width": "100%",
                                "height": "100%",
                                "borderRadius": "8px",
                                "border": "1px solid #334155",
                                "backgroundColor": "#090E1A",
                            },
                        },
                    },
                    {
                        "meta": {"name": "OptionB_WebGL_3D"},
                        "type": "ia.display.iframe",
                        "position": {"basis": "52%", "grow": 1},
                        "propConfig": {
                            "props.src": {
                                "binding": {
                                    "type": "expr",
                                    "config": {
                                        "expression": iframe_src_expr,
                                    },
                                }
                            }
                        },
                        "props": {
                            "style": {
                                "width": "100%",
                                "height": "100%",
                                "borderRadius": "8px",
                                "border": "1px solid #38BDF8",
                                "backgroundColor": "#090E1A",
                            },
                        },
                    },
                ],
            },
        ],
    }


def _build_stress_bar(
    name: str,
    label_text: str,
    value_path: str,
    min_val: float,
    max_val: float,
    bar_color: str,
) -> dict[str, Any]:
    """Construye un indicador de barra de carga física con binding absoluto a view.custom."""
    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": "50px", "shrink": 0},
        "props": {
            "direction": "column",
            "justify": "center",
            "style": {"gap": "3px", "marginBottom": "4px"},
        },
        "children": [
            {
                "meta": {"name": "Lbl_Bar"},
                "type": "ia.display.label",
                "position": {"basis": "18px"},
                "propConfig": {
                    "props.text": {
                        "binding": {
                            "type": "expr",
                            "config": {"expression": label_text},
                        }
                    }
                },
                "props": {
                    "style": {
                        "color": "#CBD5E1",
                        "fontSize": "11px",
                        "fontWeight": "600",
                    }
                },
            },
            {
                "meta": {"name": "Progress_Bar"},
                "type": "ia.display.progress",
                "position": {"basis": "15px"},
                "propConfig": {
                    "props.value": {
                        "binding": {
                            "type": "property",
                            "config": {"path": value_path},
                        }
                    }
                },
                "props": {
                    "min": min_val,
                    "max": max_val,
                    "bar": {"color": bar_color},
                    "track": {"color": "#0F172A"},
                    "style": {"borderRadius": "6px"},
                },
            },
        ],
    }


def generate_isa101_perspective_view() -> dict[str, Any]:
    """Genera la vista Turbofan_Overview con Dibujo Animado SVG de la Turbina y Simulador de Ciclos."""
    kpi_cards = [
        _build_dark_kpi_card(
            name="Card_Ciclo",
            title="Ciclo Evaluado",
            subtitle="Controlado por simulador",
            value_expression='concat({view.custom.kpis[0].current_cycle}, " vuelos")',
            accent_color="#64748B",
        ),
        _build_dark_kpi_card(
            name="Card_RUL",
            title="Vida Útil Restante (RUL IA)",
            subtitle="Modelo XGBoost (Tope 125 ciclos)",
            value_expression='concat(numberFormat({view.custom.kpis[0].rul_cycles}, "0.0"), " ciclos")',
            accent_color="#38BDF8",
        ),
        _build_dark_kpi_card(
            name="Card_IC90",
            title="Banda Conformal 90% (MAPIE)",
            subtitle="Intervalo garantizado [Min - Max]",
            value_expression=(
                'concat("[", numberFormat({view.custom.kpis[0].rul_lower_90}, "0.0"), '
                '" — ", numberFormat({view.custom.kpis[0].rul_upper_90}, "0.0"), "] cic.")'
            ),
            accent_color="#A855F7",
        ),
        _build_dark_kpi_card(
            name="Card_Ps30_s11",
            title="Presión Estática HPC (s11)",
            subtitle="Importancia IA: 52.1% | Umbral > 47.85 psia",
            value_expression='concat(numberFormat({view.custom.kpis[0].hpc_static_press_s11}, "0.00"), " psia")',
            accent_color="#0EA5E9",
        ),
        _build_dark_kpi_card(
            name="Card_Temp_s4",
            title="Temp. Salida LPT (s4)",
            subtitle="Importancia IA: 10.3% | Crítico >= 1418 °R",
            value_expression='concat(numberFormat({view.custom.kpis[0].lpt_temp_s4}, "0.0"), " °R")',
            accent_color="#F97316",
        ),
        _build_dark_kpi_card(
            name="Card_Salud",
            title="Diagnóstico Operativo",
            subtitle="Clasificación ISA-101 en vivo",
            value_binding_path="view.custom.kpis[0].health_status",
            accent_color="#22C55E",
            is_health_card=True,
        ),
    ]

    return {
        "custom": {
            "selectedAsset": "cmapss_1",
            "simCycle": 192,
        },
        "params": {},
        "propConfig": {
            "custom.kpis": {
                "binding": {
                    "type": "query",
                    "config": {
                        "parameters": {
                            "assetId": "{view.custom.selectedAsset}",
                            "simCycle": "{view.custom.simCycle}",
                        },
                        "queryPath": "CMAPSS/turbofan_kpis",
                        "returnFormat": "json",
                    },
                }
            }
        },
        "props": {"defaultSize": {"width": 1600, "height": 1120}},
        "root": {
            "meta": {"name": "root"},
            "type": "ia.container.flex",
            "props": {
                "direction": "column",
                "style": {
                    "backgroundColor": "#0B1120",
                    "color": "#F8FAFC",
                    "padding": "12px 18px",
                    "gap": "10px",
                    "fontFamily": "Inter, Segoe UI, Roboto, sans-serif",
                },
            },
            "children": [
                # 1. CABECERA DE COMANDO AEROESPACIAL
                {
                    "meta": {"name": "Header_Bar"},
                    "type": "ia.container.flex",
                    "position": {"basis": "60px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "alignItems": "center",
                        "style": {
                            "backgroundColor": "#1E293B",
                            "border": "1px solid #334155",
                            "borderRadius": "10px",
                            "padding": "8px 18px",
                            "boxShadow": "0 4px 14px rgba(0, 0, 0, 0.3)",
                        },
                    },
                    "children": [
                        {
                            "meta": {"name": "Title_Block"},
                            "type": "ia.container.flex",
                            "position": {"basis": "520px", "grow": 1},
                            "props": {"direction": "column", "justify": "center"},
                            "children": [
                                {
                                    "meta": {"name": "Lbl_Header_Title"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px"},
                                    "props": {
                                        "text": "✈️ GEMELO DIGITAL TURBOFAN — CENTRO DE CONTROL PREDICTIVO (NASA C-MAPSS)",
                                        "style": {
                                            "color": "#F8FAFC",
                                            "fontSize": "15px",
                                            "fontWeight": "800",
                                            "letterSpacing": "0.03em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Lbl_Header_Sub"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "16px"},
                                    "props": {
                                        "text": "● SIMULADOR DE DEGRADACIÓN ACTIVO | TIMESCALEDB | XGBOOST + MAPIE CONFORMAL (91% COBERTURA)",
                                        "style": {
                                            "color": "#38BDF8",
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                            "letterSpacing": "0.04em",
                                        },
                                    },
                                },
                            ],
                        },
                        {
                            "meta": {"name": "Lbl_Prescriptive_Banner"},
                            "type": "ia.display.label",
                            "position": {"basis": "360px", "shrink": 0},
                            "propConfig": {
                                "props.text": {
                                    "binding": {
                                        "type": "expr",
                                        "config": {
                                            "expression": 'concat("⚡ RECOMENDACIÓN IA: ", upper({view.custom.kpis[0].prescriptive_action}))'
                                        },
                                    }
                                },
                                "props.style.backgroundColor": {
                                    "binding": {
                                        "type": "expr",
                                        "config": {
                                            "expression": (
                                                "if({view.custom.kpis[0].prescriptive_action} = 'Parada Inmediata', '#991B1B', "
                                                "if({view.custom.kpis[0].prescriptive_action} = 'Mantenimiento Requerido', '#92400E', '#065F46'))"
                                            )
                                        },
                                    }
                                },
                                "props.style.borderColor": {
                                    "binding": {
                                        "type": "expr",
                                        "config": {
                                            "expression": (
                                                "if({view.custom.kpis[0].prescriptive_action} = 'Parada Inmediata', '#EF4444', "
                                                "if({view.custom.kpis[0].prescriptive_action} = 'Mantenimiento Requerido', '#F59E0B', '#10B981'))"
                                            )
                                        },
                                    }
                                },
                            },
                            "props": {
                                "style": {
                                    "color": "#FFFFFF",
                                    "fontSize": "12px",
                                    "fontWeight": "800",
                                    "padding": "7px 14px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #EF4444",
                                    "textAlign": "center",
                                    "marginRight": "18px",
                                    "letterSpacing": "0.03em",
                                }
                            },
                        },
                        {
                            "meta": {"name": "Nav_To_Guide"},
                            "type": "ia.navigation.link",
                            "position": {"basis": "275px", "shrink": 0},
                            "props": {
                                "page": "/guia",
                                "text": "🎓 GUÍA ARQUITECTURA, NQ & MCP",
                                "style": {
                                    "backgroundColor": "#1E1B4B",
                                    "color": "#C084FC",
                                    "border": "1px solid #A855F7",
                                    "borderRadius": "8px",
                                    "padding": "7px 12px",
                                    "fontWeight": "800",
                                    "fontSize": "12px",
                                    "textAlign": "center",
                                    "textDecoration": "none",
                                    "marginRight": "12px",
                                    "letterSpacing": "0.03em",
                                },
                            },
                        },
                        {
                            "meta": {"name": "Dropdown"},
                            "type": "ia.input.dropdown",
                            "position": {"basis": "220px", "shrink": 0},
                            "propConfig": {
                                "props.options": {
                                    "binding": {
                                        "type": "query",
                                        "config": {
                                            "queryPath": "CMAPSS/active_assets",
                                            "returnFormat": "json",
                                        },
                                    }
                                },
                                "props.value": {
                                    "binding": {
                                        "type": "property",
                                        "config": {
                                            "bidirectional": True,
                                            "path": "view.custom.selectedAsset",
                                        },
                                    }
                                },
                            },
                            "props": {
                                "style": {
                                    "backgroundColor": "#0F172A",
                                    "color": "#F8FAFC",
                                    "border": "1px solid #38BDF8",
                                    "borderRadius": "8px",
                                    "fontWeight": "700",
                                    "fontSize": "14px",
                                },
                            },
                        },
                    ],
                },
                # 2. FILA DE 6 TARJETAS KPI DARK-MODE
                {
                    "meta": {"name": "KPI_Row"},
                    "type": "ia.container.flex",
                    "position": {"basis": "98px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "wrap": "wrap",
                    },
                    "children": kpi_cards,
                },
                # 3. DIBUJO VECTORIAL ANIMADO DEL TURBOFAN + SIMULADOR DE CICLO DE VUELO
                _build_turbofan_synoptic_section(),
                # 4. CUERPO PRINCIPAL EN 3 COLUMNAS: [PANEL IA + GAUGE] | [CURVA TERMODINÁMICA] | [TABLA 24 SENSORES]
                {
                    "meta": {"name": "Main_Content_Row"},
                    "type": "ia.container.flex",
                    "position": {"basis": "520px", "grow": 1},
                    "props": {
                        "direction": "row",
                        "gap": "12px",
                    },
                    "children": [
                        # COLUMNA 1 (25%): GAUGE SEMICIRCULAR DE RUL + BARRAS DE ESTRÉS FÍSICO IA
                        {
                            "meta": {"name": "AI_Diagnostics_Panel"},
                            "type": "ia.container.flex",
                            "position": {"basis": "25%", "grow": 1},
                            "props": {
                                "direction": "column",
                                "justify": "space-between",
                                "style": {
                                    "backgroundColor": "#1E293B",
                                    "borderRadius": "10px",
                                    "border": "1px solid #334155",
                                    "padding": "12px 14px",
                                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                                },
                            },
                            "children": [
                                {
                                    "meta": {"name": "Lbl_Gauge_Header"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "20px", "shrink": 0},
                                    "props": {
                                        "text": "MEDIDOR DE VIDA ÚTIL (RUL GAUGE)",
                                        "style": {
                                            "color": "#94A3B8",
                                            "fontSize": "11px",
                                            "fontWeight": "700",
                                            "letterSpacing": "0.05em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Gauge_RUL"},
                                    "type": "ia.chart.simple-gauge",
                                    "position": {"basis": "175px", "grow": 1},
                                    "propConfig": {
                                        "props.value": {
                                            "binding": {
                                                "type": "property",
                                                "config": {
                                                    "path": "view.custom.kpis[0].rul_cycles"
                                                },
                                            }
                                        },
                                        "props.arc.color": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": (
                                                        "if({view.custom.kpis[0].rul_cycles} <= 25, '#EF4444', "
                                                        "if({view.custom.kpis[0].rul_cycles} <= 60, '#F59E0B', '#10B981'))"
                                                    )
                                                },
                                            }
                                        },
                                    },
                                    "props": {
                                        "minValue": 0,
                                        "maxValue": 125,
                                        "arc": {"width": 20, "backgroundColor": "#0F172A"},
                                        "label": {"size": 22, "color": "#F8FAFC", "units": " cic"},
                                    },
                                },
                                {
                                    "meta": {"name": "Lbl_Stress_Header"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "20px", "shrink": 0},
                                    "props": {
                                        "text": "ÍNDICES DE ESTRÉS FÍSICO (TOP SENSORES IA)",
                                        "style": {
                                            "color": "#94A3B8",
                                            "fontSize": "11px",
                                            "fontWeight": "700",
                                            "letterSpacing": "0.05em",
                                        },
                                    },
                                },
                                _build_stress_bar(
                                    name="Bar_s11",
                                    label_text='concat("s11 Presión HPC (52% IA): ", numberFormat({view.custom.kpis[0].hpc_static_press_s11}, "0.00"), " / 48.5 psia")',
                                    value_path="view.custom.kpis[0].hpc_static_press_s11",
                                    min_val=46.8,
                                    max_val=48.5,
                                    bar_color="#38BDF8",
                                ),
                                _build_stress_bar(
                                    name="Bar_s4",
                                    label_text='concat("s4 Temp. Salida LPT (10% IA): ", numberFormat({view.custom.kpis[0].lpt_temp_s4}, "0.0"), " / 1440 °R")',
                                    value_path="view.custom.kpis[0].lpt_temp_s4",
                                    min_val=1385.0,
                                    max_val=1440.0,
                                    bar_color="#F97316",
                                ),
                                _build_stress_bar(
                                    name="Bar_s9",
                                    label_text='concat("s9 Velocidad Núcleo Nc (12% IA): ", numberFormat({view.custom.kpis[0].core_speed_s9}, "0.0"), " / 9200 rpm")',
                                    value_path="view.custom.kpis[0].core_speed_s9",
                                    min_val=9020.0,
                                    max_val=9200.0,
                                    bar_color="#A855F7",
                                ),
                            ],
                        },
                        # COLUMNA 2 (45%): GRÁFICA DE TENDENCIA TERMODINÁMICA HASTA EL CICLO SIMULADO
                        {
                            "meta": {"name": "Chart_Container"},
                            "type": "ia.container.flex",
                            "position": {"basis": "45%", "grow": 2},
                            "props": {
                                "direction": "column",
                                "style": {
                                    "backgroundColor": "#1E293B",
                                    "borderRadius": "10px",
                                    "border": "1px solid #334155",
                                    "padding": "12px 14px",
                                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                                },
                            },
                            "children": [
                                {
                                    "meta": {"name": "Lbl_Chart_Header"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px", "shrink": 0},
                                    "props": {
                                        "text": "EVOLUCIÓN TÉRMICA HASTA EL CICLO SIMULADO (LPT s4 & HPC s3 [°R])",
                                        "style": {
                                            "color": "#F8FAFC",
                                            "fontSize": "12px",
                                            "fontWeight": "700",
                                            "letterSpacing": "0.04em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "TimeSeriesChart"},
                                    "type": "ia.chart.timeseries",
                                    "position": {"basis": "450px", "grow": 1},
                                    "propConfig": {
                                        "props.series[0].data": {
                                            "binding": {
                                                "type": "query",
                                                "config": {
                                                    "parameters": {
                                                        "assetId": "{view.custom.selectedAsset}",
                                                        "startCycle": "1",
                                                        "endCycle": "{view.custom.simCycle}",
                                                    },
                                                    "queryPath": "CMAPSS/telemetry_trend",
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "series": [{"name": "DegradacionTermica"}],
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "borderRadius": "8px",
                                            "padding": "8px",
                                        },
                                    },
                                },
                            ],
                        },
                        # COLUMNA 3 (30%): TABLA DE INSTRUMENTACIÓN DE 24 SENSORES
                        {
                            "meta": {"name": "Table_Container"},
                            "type": "ia.container.flex",
                            "position": {"basis": "30%", "grow": 1},
                            "props": {
                                "direction": "column",
                                "style": {
                                    "backgroundColor": "#1E293B",
                                    "borderRadius": "10px",
                                    "border": "1px solid #334155",
                                    "padding": "12px 14px",
                                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                                },
                            },
                            "children": [
                                {
                                    "meta": {"name": "Lbl_Table_Header"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px", "shrink": 0},
                                    "props": {
                                        "text": "SNAPSHOT DE INSTRUMENTACIÓN (24 SEÑALES)",
                                        "style": {
                                            "color": "#F8FAFC",
                                            "fontSize": "12px",
                                            "fontWeight": "700",
                                            "letterSpacing": "0.04em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Table"},
                                    "type": "ia.display.table",
                                    "position": {"basis": "450px", "grow": 1},
                                    "propConfig": {
                                        "props.data": {
                                            "binding": {
                                                "type": "query",
                                                "config": {
                                                    "parameters": {
                                                        "assetId": "{view.custom.selectedAsset}"
                                                    },
                                                    "queryPath": "CMAPSS/latest_sensors",
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "pager": {"bottom": False},
                                        "rows": {"striped": True},
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "color": "#E2E8F0",
                                            "borderRadius": "8px",
                                            "fontSize": "12px",
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    }


def _build_architecture_pipeline_svg_expr() -> str:
    """
    Construye una expresión Perspective que genera un Diagrama SVG Animado de Arquitectura End-to-End
    + Gráfica de Barras SVG de Importancia de Sensores IA (XGBoost), usando exclusivamente rgb() sin '#'.
    """
    parts = [
        '"data:image/svg+xml;utf8,',
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1500 250' width='100%' height='100%'>",
        "<rect width='1500' height='250' rx='10' fill='rgb(9,14,26)'/>",
        # Cuadrícula técnica de fondo
        "<g stroke='rgb(30,41,59)' stroke-width='1' opacity='0.55'>",
        "<line x1='0' y1='50' x2='1500' y2='50'/><line x1='0' y1='125' x2='1500' y2='125'/><line x1='0' y1='200' x2='1500' y2='200'/>",
        "<line x1='250' y1='0' x2='250' y2='250'/><line x1='500' y1='0' x2='500' y2='250'/><line x1='750' y1='0' x2='750' y2='250'/>",
        "<line x1='1000' y1='0' x2='1000' y2='250'/><line x1='1250' y1='0' x2='1250' y2='250'/>",
        "</g>",
        # Título superior del diagrama SVG
        "<text x='20' y='26' fill='rgb(56,189,248)' font-family='sans-serif' font-size='13' font-weight='bold'>",
        "FLUJO DE DATOS EN TIEMPO REAL — ARQUITECTURA END-TO-END DEL GEMELO DIGITAL + SERVIDOR MCP + PERSPECTIVE 8.3",
        "</text>",
        "<text x='1480' y='26' text-anchor='end' fill='rgb(168,85,247)' font-family='monospace' font-size='12' font-weight='bold'>",
        "ACTIVO EN PRUEBA: \" + {view.custom.selectedAsset} + \" | CICLO: \" + {view.custom.simCycle} + \" | RUL IA: \" + numberFormat( coalesce({view.custom.kpis[0].rul_cycles}, 0.0), \"0.0\" ) + \" CICLOS",
        "</text>",
        # Tuberías de conexión entre los 5 bloques de arquitectura
        "<g stroke='rgb(51,65,85)' stroke-width='4' fill='none'>",
        "<line x1='215' y1='115' x2='255' y2='115'/>",
        "<line x1='455' y1='115' x2='495' y2='115'/>",
        "<line x1='695' y1='115' x2='735' y2='115'/>",
        "<line x1='945' y1='115' x2='985' y2='115'/>",
        "</g>",
        # Paquetes de datos animados circulando por las tuberías
        "<circle r='5' fill='rgb(56,189,248)'><animate attributeName='cx' values='215;255' dur='0.9s' repeatCount='indefinite'/><animate attributeName='cy' values='115;115' dur='0.9s' repeatCount='indefinite'/></circle>",
        "<circle r='5' fill='rgb(16,185,129)'><animate attributeName='cx' values='455;495' dur='0.9s' repeatCount='indefinite'/><animate attributeName='cy' values='115;115' dur='0.9s' repeatCount='indefinite'/></circle>",
        "<circle r='5' fill='rgb(245,158,11)'><animate attributeName='cx' values='695;735' dur='0.9s' repeatCount='indefinite'/><animate attributeName='cy' values='115;115' dur='0.9s' repeatCount='indefinite'/></circle>",
        "<circle r='5' fill='rgb(168,85,247)'><animate attributeName='cx' values='945;985' dur='0.9s' repeatCount='indefinite'/><animate attributeName='cy' values='115;115' dur='0.9s' repeatCount='indefinite'/></circle>",
        # BLOQUE 1: NASA C-MAPSS + MQTT
        "<rect x='18' y='45' width='197' height='145' rx='8' fill='rgb(15,23,42)' stroke='rgb(56,189,248)' stroke-width='2'/>",
        "<rect x='18' y='45' width='197' height='26' rx='8' fill='rgb(12,74,110)'/>",
        "<text x='116' y='62' text-anchor='middle' fill='rgb(255,255,255)' font-family='sans-serif' font-size='11' font-weight='bold'>1. EDGE &amp; NASA C-MAPSS</text>",
        "<text x='28' y='90' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>datasets/cmapss/adapter.py</text>",
        "<text x='28' y='110' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• 21 Sensores + 3 OpSettings</text>",
        "<text x='28' y='128' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• Tope RUL Piecewise: 125 cic</text>",
        "<text x='28' y='146' fill='rgb(56,189,248)' font-family='monospace' font-size='10'>• Mosquitto MQTT :1883</text>",
        "<text x='28' y='172' fill='rgb(16,185,129)' font-family='monospace' font-size='10' font-weight='bold'>● 24 Señales / Ciclo</text>",
        # BLOQUE 2: TIMESCALEDB + IA XGBOOST/MAPIE
        "<rect x='255' y='45' width='200' height='145' rx='8' fill='rgb(15,23,42)' stroke='rgb(16,185,129)' stroke-width='2'/>",
        "<rect x='255' y='45' width='200' height='26' rx='8' fill='rgb(6,95,70)'/>",
        "<text x='355' y='62' text-anchor='middle' fill='rgb(255,255,255)' font-family='sans-serif' font-size='11' font-weight='bold'>2. TIMESCALEDB + IA RUL</text>",
        "<text x='265' y='90' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>PostgreSQL 16 (:5432)</text>",
        "<text x='265' y='110' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• Hypertables: telemetry/pred</text>",
        "<text x='265' y='128' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• XGBoost (RMSE: 14.12 cic)</text>",
        "<text x='265' y='146' fill='rgb(16,185,129)' font-family='monospace' font-size='10'>• MAPIE Conformal 90% (91%)</text>",
        "<text x='265' y='172' fill='rgb(56,189,248)' font-family='monospace' font-size='10' font-weight='bold'>IC90: [\" + numberFormat( coalesce({view.custom.kpis[0].rul_lower_90}, 0.0), \"0.0\" ) + \" - \" + numberFormat( coalesce({view.custom.kpis[0].rul_upper_90}, 0.0), \"0.0\" ) + \"]</text>",
        # BLOQUE 3: NAMED QUERIES 8.3
        "<rect x='495' y='45' width='200' height='145' rx='8' fill='rgb(15,23,42)' stroke='rgb(245,158,11)' stroke-width='2'/>",
        "<rect x='495' y='45' width='200' height='26' rx='8' fill='rgb(146,64,14)'/>",
        "<text x='595' y='62' text-anchor='middle' fill='rgb(255,255,255)' font-family='sans-serif' font-size='11' font-weight='bold'>3. NAMED QUERIES 8.3</text>",
        "<text x='505' y='90' fill='rgb(253,230,138)' font-family='monospace' font-size='10'>CMAPSS/turbofan_kpis</text>",
        "<text x='505' y='110' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• query.sql + resource.json</text>",
        "<text x='505' y='128' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• returnFormat: json</text>",
        "<text x='505' y='146' fill='rgb(245,158,11)' font-family='monospace' font-size='10'>• Params en view.json: STRING</text>",
        "<text x='505' y='172' fill='rgb(255,255,255)' font-family='monospace' font-size='10' font-weight='bold'>:assetId='\" + {view.custom.selectedAsset} + \"'</text>",
        # BLOQUE 4: ESTADO GLOBAL view.custom
        "<rect x='735' y='45' width='210' height='145' rx='8' fill='rgb(15,23,42)' stroke='rgb(168,85,247)' stroke-width='2'/>",
        "<rect x='735' y='45' width='210' height='26' rx='8' fill='rgb(88,28,135)'/>",
        "<text x='840' y='62' text-anchor='middle' fill='rgb(255,255,255)' font-family='sans-serif' font-size='11' font-weight='bold'>4. ESTADO GLOBAL view.custom</text>",
        "<text x='745' y='90' fill='rgb(233,213,255)' font-family='monospace' font-size='10'>{view.custom.kpis[0].*}</text>",
        "<text x='745' y='110' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• Cero rutas relativas rotas</text>",
        "<text x='745' y='128' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>• Bidirectional Slider/Drop</text>",
        "<text x='745' y='146' fill='rgb(56,189,248)' font-family='monospace' font-size='10'>• SVG rgb() + 3D WebGL URL-Hash</text>",
        "<text x='745' y='172' fill='rgb(16,185,129)' font-family='monospace' font-size='10' font-weight='bold'>Salud: \" + coalesce({view.custom.kpis[0].health_status}, \"Normal\") + \"</text>",
        # BLOQUE 5 (DERECHA AMPLIA): GRÁFICA SVG DE BARRAS DE IMPORTANCIA IA + SERVIDOR MCP
        "<rect x='985' y='45' width='497' height='185' rx='8' fill='rgb(15,23,42)' stroke='rgb(56,189,248)' stroke-width='2'/>",
        "<rect x='985' y='45' width='497' height='26' rx='8' fill='rgb(30,41,59)'/>",
        "<text x='1233' y='62' text-anchor='middle' fill='rgb(56,189,248)' font-family='sans-serif' font-size='11' font-weight='bold'>5. IMPORTANCIA DE SENSORES IA (XGBOOST) &amp; SERVIDOR FASTMCP (6 TOOLS)</text>",
        # Barras horizontales de Feature Importance
        "<text x='1000' y='92' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>s11 Presión HPC (52.1%)</text>",
        "<rect x='1165' y='81' width='230' height='13' rx='3' fill='rgb(30,41,59)'/>",
        "<rect x='1165' y='81' width='195' height='13' rx='3' fill='rgb(239,68,68)'/>",
        "<text x='1405' y='92' fill='rgb(248,113,113)' font-family='monospace' font-size='10' font-weight='bold'>\" + numberFormat( coalesce({view.custom.kpis[0].hpc_static_press_s11}, 47.5), \"0.00\" ) + \" psia</text>",
        "<text x='1000' y='114' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>s9  Vel. Núcleo (11.7%)</text>",
        "<rect x='1165' y='103' width='230' height='13' rx='3' fill='rgb(30,41,59)'/>",
        "<rect x='1165' y='103' width='68' height='13' rx='3' fill='rgb(168,85,247)'/>",
        "<text x='1405' y='114' fill='rgb(192,132,252)' font-family='monospace' font-size='10' font-weight='bold'>\" + numberFormat( coalesce({view.custom.kpis[0].core_speed_s9}, 9050.0), \"0.0\" ) + \" rpm</text>",
        "<text x='1000' y='136' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>s4  Temp. LPT   (10.3%)</text>",
        "<rect x='1165' y='125' width='230' height='13' rx='3' fill='rgb(30,41,59)'/>",
        "<rect x='1165' y='125' width='60' height='13' rx='3' fill='rgb(249,115,22)'/>",
        "<text x='1405' y='136' fill='rgb(251,146,60)' font-family='monospace' font-size='10' font-weight='bold'>\" + numberFormat( coalesce({view.custom.kpis[0].lpt_temp_s4}, 1400.0), \"0.0\" ) + \" °R</text>",
        "<text x='1000' y='158' fill='rgb(226,232,240)' font-family='monospace' font-size='10'>s12 Flujo Comb.  (5.2%)</text>",
        "<rect x='1165' y='147' width='230' height='13' rx='3' fill='rgb(30,41,59)'/>",
        "<rect x='1165' y='147' width='34' height='13' rx='3' fill='rgb(56,189,248)'/>",
        "<text x='1405' y='158' fill='rgb(56,189,248)' font-family='monospace' font-size='10' font-weight='bold'>\" + numberFormat( coalesce({view.custom.kpis[0].fuel_flow_s12}, 521.0), \"0.0\" ) + \" pps</text>",
        "<text x='1000' y='182' fill='rgb(16,185,129)' font-family='monospace' font-size='10' font-weight='bold'>MCP Tools (src/mcp_server.py): get_fleet_status | predict_engine_rul | audit_hmi | deploy_hmi</text>",
        "<text x='1000' y='202' fill='rgb(148,163,184)' font-family='sans-serif' font-size='10'>Mueve el Slider o cambia de Turbina arriba: este SVG, el Gemelo 3D y las Tablas reaccionan al instante.</text>",
        # Pie informativo izquierdo
        "<rect x='18' y='200' width='927' height='30' rx='6' fill='rgb(15,23,42)' stroke='rgb(51,65,85)' stroke-width='1'/>",
        "<text x='30' y='219' fill='rgb(226,232,240)' font-family='monospace' font-size='11'>",
        "💡 REGLA DE ORO PERSPECTIVE 8.3: Dropdown/Slider ➔ view.custom.selectedAsset / simCycle ➔ Named Query (CMAPSS/turbofan_kpis) ➔ view.custom.kpis[0]",
        "</text>",
        "</svg>\"",
    ]
    return "".join(parts)


def _build_guide_tech_card(
    name: str,
    badge_text: str,
    title: str,
    bullet_1: str,
    bullet_2: str,
    code_snippet: str,
    accent_color: str,
) -> dict[str, Any]:
    """Construye una tarjeta pedagógica para explicar conceptos clave de Ignition Perspective 8.3 y MCP."""
    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": "24%", "grow": 1},
        "props": {
            "direction": "column",
            "justify": "space-between",
            "style": {
                "backgroundColor": "#1E293B",
                "borderRadius": "10px",
                "border": "1px solid #334155",
                "borderTop": f"4px solid {accent_color}",
                "padding": "10px 14px",
                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                "gap": "4px",
            },
        },
        "children": [
            {
                "meta": {"name": "Lbl_Badge"},
                "type": "ia.display.label",
                "position": {"basis": "16px", "shrink": 0},
                "props": {
                    "text": badge_text,
                    "style": {
                        "color": accent_color,
                        "fontSize": "10px",
                        "fontWeight": "800",
                        "letterSpacing": "0.06em",
                    },
                },
            },
            {
                "meta": {"name": "Lbl_Title"},
                "type": "ia.display.label",
                "position": {"basis": "20px", "shrink": 0},
                "props": {
                    "text": title,
                    "style": {
                        "color": "#F8FAFC",
                        "fontSize": "13px",
                        "fontWeight": "800",
                    },
                },
            },
            {
                "meta": {"name": "Lbl_B1"},
                "type": "ia.display.label",
                "position": {"basis": "32px", "shrink": 0},
                "props": {
                    "text": bullet_1,
                    "style": {
                        "color": "#CBD5E1",
                        "fontSize": "11px",
                        "lineHeight": "1.3",
                    },
                },
            },
            {
                "meta": {"name": "Lbl_B2"},
                "type": "ia.display.label",
                "position": {"basis": "32px", "shrink": 0},
                "props": {
                    "text": bullet_2,
                    "style": {
                        "color": "#94A3B8",
                        "fontSize": "11px",
                        "lineHeight": "1.3",
                    },
                },
            },
            {
                "meta": {"name": "Lbl_Code"},
                "type": "ia.display.label",
                "position": {"basis": "26px", "shrink": 0},
                "props": {
                    "text": code_snippet,
                    "style": {
                        "backgroundColor": "#090E1A",
                        "color": "#38BDF8",
                        "fontFamily": "JetBrains Mono, Consolas, monospace",
                        "fontSize": "10px",
                        "fontWeight": "700",
                        "padding": "4px 8px",
                        "borderRadius": "6px",
                        "border": "1px solid #1E293B",
                    },
                },
            },
        ],
    }


def generate_architecture_guide_view() -> dict[str, Any]:
    """
    Genera la vista educativa e interactiva `Arquitectura_Guia` (URL `/guia`) en Ignition Perspective 8.3.
    Explica Named Queries, `view.custom`, SVG reactivos, Gemelo 3D en Despiece (Exploded View) y el Servidor MCP.
    """
    guia_iframe_expr = (
        'concat("http://localhost:8095/guia_gemelo3d.html#asset=", {view.custom.selectedAsset}, '
        '"&cycle=", {view.custom.simCycle}, '
        '"&rul=", coalesce({view.custom.kpis[0].rul_cycles}, 7.7), '
        '"&s4=", coalesce({view.custom.kpis[0].lpt_temp_s4}, 1425.8), '
        '"&s11=", coalesce({view.custom.kpis[0].hpc_static_press_s11}, 48.18), '
        '"&s9=", coalesce({view.custom.kpis[0].core_speed_s9}, 9062.0), '
        '"&health=", coalesce({view.custom.kpis[0].health_status}, "Critico"))'
    )

    tech_cards = [
        _build_guide_tech_card(
            name="Card_NQ",
            badge_text="1. NAMED QUERIES EN IGNITION 8.3",
            title="Carpeta query.sql + resource.json",
            bullet_1="• Cada Named Query vive en ignition/named-query/CMAPSS/<nombre>/ con su SQL y su resource.json (sqlType 7=String, 2=Int4).",
            bullet_2="• CRÍTICO EN 8.3: En view.json, todos los parámetros en binding.config.parameters deben escribirse como STRING (\"1\", no 1).",
            code_snippet='parameters: {"assetId": "{view.custom.selectedAsset}", "startCycle": "1"}',
            accent_color="#F59E0B",
        ),
        _build_guide_tech_card(
            name="Card_ViewCustom",
            badge_text="2. ADIÓS A RUTAS RELATIVAS ANIDADAS",
            title="Patrón de Estado Global view.custom",
            bullet_1="• Antes se usaban rutas relativas ({.. / Header / Dropdown.props.value}) que se rompían al mover contenedores Flex.",
            bullet_2="• Ahora Dropdown y Slider usan bidirectional: true hacia view.custom, y 1 solo Named Query llena view.custom.kpis[0].",
            code_snippet="Dropdown (bidirectional) ➔ view.custom.selectedAsset ➔ view.custom.kpis[0].rul_cycles",
            accent_color="#A855F7",
        ),
        _build_guide_tech_card(
            name="Card_SVG",
            badge_text="3. DIBUJOS CAD SVG EN PERSPECTIVE",
            title="Data URI + Animación Nativa SMIL",
            bullet_1="• Puedes inyectar código <svg> dinámico en un componente ia.display.image usando concat('data:image/svg+xml;utf8,<svg>...').",
            bullet_2="• REGLA CRÍTICA: Nunca uses '#' en colores dentro del Data URI porque el navegador corta el texto; usa siempre rgb(r,g,b).",
            code_snippet="<animateTransform type='rotate' from='0 240 120' to='360 240 120' dur='0.4s'/>",
            accent_color="#38BDF8",
        ),
        _build_guide_tech_card(
            name="Card_3D_MCP",
            badge_text="4. GEMELO 3D WEBGL + SERVIDOR FASTMCP",
            title="Sincronía #hash a 60 FPS e IA Agéntica",
            bullet_1="• El visor Three.js recibe telemetría por URL #hash (window.onhashchange) a 60 FPS sin recargar el iframe ni perder la cámara.",
            bullet_2="• src/mcp_server.py expone 6 herramientas FastMCP para auditar vistas Perspective, predecir RUL con XGBoost+MAPIE y desplegar.",
            code_snippet="python src/mcp_server.py --demo  |  6 Tools MCP listas para Claude/Cursor",
            accent_color="#10B981",
        ),
    ]

    return {
        "custom": {
            "selectedAsset": "cmapss_1",
            "simCycle": 192,
        },
        "params": {},
        "propConfig": {
            "custom.kpis": {
                "binding": {
                    "type": "query",
                    "config": {
                        "parameters": {
                            "assetId": "{view.custom.selectedAsset}",
                            "simCycle": "{view.custom.simCycle}",
                        },
                        "queryPath": "CMAPSS/turbofan_kpis",
                        "returnFormat": "json",
                    },
                }
            }
        },
        "props": {"defaultSize": {"width": 1600, "height": 1240}},
        "root": {
            "meta": {"name": "root"},
            "type": "ia.container.flex",
            "props": {
                "direction": "column",
                "style": {
                    "backgroundColor": "#0B1120",
                    "color": "#F8FAFC",
                    "padding": "12px 18px",
                    "gap": "10px",
                    "fontFamily": "Inter, Segoe UI, Roboto, sans-serif",
                },
            },
            "children": [
                # 1. CABECERA CON CONTROLES EN VIVO Y BOTÓN PARA VOLVER AL HMI OPERATIVO
                {
                    "meta": {"name": "Header_Guia"},
                    "type": "ia.container.flex",
                    "position": {"basis": "62px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "alignItems": "center",
                        "style": {
                            "backgroundColor": "#1E293B",
                            "border": "1px solid #38BDF8",
                            "borderRadius": "10px",
                            "padding": "8px 16px",
                            "boxShadow": "0 4px 14px rgba(0, 0, 0, 0.35)",
                        },
                    },
                    "children": [
                        {
                            "meta": {"name": "Title_Block"},
                            "type": "ia.container.flex",
                            "position": {"basis": "520px", "grow": 1},
                            "props": {"direction": "column", "justify": "center"},
                            "children": [
                                {
                                    "meta": {"name": "Lbl_Title"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px"},
                                    "props": {
                                        "text": "🎓 CENTRO DE ARQUITECTURA E INGENIERÍA: PERSPECTIVE 8.3 + NAMED QUERIES + MCP + GEMELO 3D",
                                        "style": {
                                            "color": "#F8FAFC",
                                            "fontSize": "14px",
                                            "fontWeight": "800",
                                            "letterSpacing": "0.03em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Lbl_Sub"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "16px"},
                                    "props": {
                                        "text": "● GUÍA INTERACTIVA EN VIVO | PRUEBA LOS CONTROLES AQUÍ MISMO PARA VER CÓMO RESPONDEN LOS NAMED QUERIES Y EL GEMELO 3D",
                                        "style": {
                                            "color": "#38BDF8",
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                        },
                                    },
                                },
                            ],
                        },
                        {
                            "meta": {"name": "Lbl_Cycle_Guia"},
                            "type": "ia.display.label",
                            "position": {"basis": "165px", "shrink": 0},
                            "propConfig": {
                                "props.text": {
                                    "binding": {
                                        "type": "expr",
                                        "config": {
                                            "expression": 'concat("⏱️ Ciclo Simulado: ", {view.custom.simCycle})'
                                        },
                                    }
                                }
                            },
                            "props": {
                                "style": {
                                    "color": "#38BDF8",
                                    "fontSize": "12px",
                                    "fontWeight": "800",
                                    "textAlign": "right",
                                    "marginRight": "8px",
                                }
                            },
                        },
                        {
                            "meta": {"name": "Slider_Guia"},
                            "type": "ia.input.slider",
                            "position": {"basis": "200px", "shrink": 0},
                            "propConfig": {
                                "props.value": {
                                    "binding": {
                                        "type": "property",
                                        "config": {
                                            "bidirectional": True,
                                            "path": "view.custom.simCycle",
                                        },
                                    }
                                }
                            },
                            "props": {
                                "min": 1,
                                "max": 192,
                                "step": 1,
                                "color": "#38BDF8",
                                "style": {"marginRight": "12px"},
                            },
                        },
                        {
                            "meta": {"name": "Dropdown_Guia"},
                            "type": "ia.input.dropdown",
                            "position": {"basis": "165px", "shrink": 0},
                            "propConfig": {
                                "props.options": {
                                    "binding": {
                                        "type": "query",
                                        "config": {
                                            "queryPath": "CMAPSS/active_assets",
                                            "returnFormat": "json",
                                        },
                                    }
                                },
                                "props.value": {
                                    "binding": {
                                        "type": "property",
                                        "config": {
                                            "bidirectional": True,
                                            "path": "view.custom.selectedAsset",
                                        },
                                    }
                                },
                            },
                            "props": {
                                "style": {
                                    "backgroundColor": "#0F172A",
                                    "color": "#F8FAFC",
                                    "border": "1px solid #38BDF8",
                                    "borderRadius": "8px",
                                    "fontWeight": "700",
                                    "fontSize": "13px",
                                    "marginRight": "12px",
                                }
                            },
                        },
                        {
                            "meta": {"name": "Nav_Back_HMI"},
                            "type": "ia.navigation.link",
                            "position": {"basis": "235px", "shrink": 0},
                            "props": {
                                "page": "/",
                                "text": "⬅️ VOLVER AL HMI OPERATIVO",
                                "style": {
                                    "backgroundColor": "#065F46",
                                    "color": "#FFFFFF",
                                    "border": "1px solid #10B981",
                                    "borderRadius": "8px",
                                    "padding": "8px 12px",
                                    "fontWeight": "800",
                                    "fontSize": "12px",
                                    "textAlign": "center",
                                    "textDecoration": "none",
                                    "letterSpacing": "0.03em",
                                },
                            },
                        },
                    ],
                },
                # 2. DIAGRAMA SVG ANIMADO DE ARQUITECTURA END-TO-END + BARRAS DE IMPORTANCIA IA
                {
                    "meta": {"name": "Arch_SVG_Container"},
                    "type": "ia.container.flex",
                    "position": {"basis": "240px", "shrink": 0},
                    "props": {
                        "direction": "column",
                        "style": {
                            "backgroundColor": "#1E293B",
                            "borderRadius": "10px",
                            "border": "1px solid #334155",
                            "padding": "6px 10px",
                            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.25)",
                        },
                    },
                    "children": [
                        {
                            "meta": {"name": "Arch_Pipeline_SVG"},
                            "type": "ia.display.image",
                            "position": {"basis": "224px", "grow": 1},
                            "propConfig": {
                                "props.source": {
                                    "binding": {
                                        "type": "expr",
                                        "config": {
                                            "expression": _build_architecture_pipeline_svg_expr(),
                                        },
                                    }
                                }
                            },
                            "props": {
                                "fit": {"mode": "contain"},
                                "style": {
                                    "width": "100%",
                                    "height": "100%",
                                    "borderRadius": "8px",
                                    "backgroundColor": "#090E1A",
                                },
                            },
                        }
                    ],
                },
                # 3. LAS 4 TARJETAS DE ACTUALIZACIÓN TÉCNICA (NAMED QUERIES, VIEW.CUSTOM, SVG, 3D + MCP)
                {
                    "meta": {"name": "Tech_Cards_Row"},
                    "type": "ia.container.flex",
                    "position": {"basis": "165px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "style": {"gap": "12px"},
                    },
                    "children": tech_cards,
                },
                # 4. LABORATORIO INTERACTIVO: GEMELO 3D EN DESPIECE (EXPLODED VIEW) + INSPECTOR EN VIVO DE NAMED QUERIES
                {
                    "meta": {"name": "Interactive_Lab_Row"},
                    "type": "ia.container.flex",
                    "position": {"basis": "690px", "grow": 1},
                    "props": {
                        "direction": "row",
                        "style": {"gap": "12px"},
                    },
                    "children": [
                        # MITAD IZQUIERDA (60%): GEMELO 3D CON DESPIECE MECÁNICO (EXPLODED VIEW) + 5 PESTAÑAS DE CÓDIGO
                        {
                            "meta": {"name": "Exploded_3D_Guide_Panel"},
                            "type": "ia.container.flex",
                            "position": {"basis": "60%", "grow": 1},
                            "props": {
                                "direction": "column",
                                "style": {
                                    "backgroundColor": "#1E293B",
                                    "borderRadius": "10px",
                                    "border": "1px solid #38BDF8",
                                    "padding": "10px 12px",
                                    "gap": "6px",
                                },
                            },
                            "children": [
                                {
                                    "meta": {"name": "Lbl_3D_Lab_Title"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px", "shrink": 0},
                                    "props": {
                                        "text": "🔧 LABORATORIO 3D EN DESPIECE MECÁNICO (EXPLODED VIEW) + MANUAL INTERACTIVO EN 5 PESTAÑAS (PULSA '💥 DESPIECE MECÁNICO 3D')",
                                        "style": {
                                            "color": "#38BDF8",
                                            "fontSize": "12px",
                                            "fontWeight": "800",
                                            "letterSpacing": "0.03em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Iframe_Guia_3D"},
                                    "type": "ia.display.iframe",
                                    "position": {"basis": "640px", "grow": 1},
                                    "propConfig": {
                                        "props.src": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": guia_iframe_expr,
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "style": {
                                            "width": "100%",
                                            "height": "100%",
                                            "borderRadius": "8px",
                                            "border": "1px solid #334155",
                                            "backgroundColor": "#090E1A",
                                        },
                                    },
                                },
                            ],
                        },
                        # MITAD DERECHA (40%): INSPECTOR EN VIVO DEL JSON DE NAMED QUERIES + GRÁFICA DE TENDENCIA
                        {
                            "meta": {"name": "NQ_Live_Inspector_Panel"},
                            "type": "ia.container.flex",
                            "position": {"basis": "40%", "grow": 1},
                            "props": {
                                "direction": "column",
                                "style": {
                                    "backgroundColor": "#1E293B",
                                    "borderRadius": "10px",
                                    "border": "1px solid #334155",
                                    "padding": "10px 12px",
                                    "gap": "8px",
                                },
                            },
                            "children": [
                                {
                                    "meta": {"name": "Lbl_NQ_Inspector_Title"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px", "shrink": 0},
                                    "props": {
                                        "text": "🔍 INSPECTOR EN VIVO: SALIDA JSON DEL NAMED QUERY (CMAPSS/turbofan_kpis)",
                                        "style": {
                                            "color": "#F59E0B",
                                            "fontSize": "12px",
                                            "fontWeight": "800",
                                            "letterSpacing": "0.03em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Lbl_NQ_Inspector_Sub"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "32px", "shrink": 0},
                                    "propConfig": {
                                        "props.text": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": (
                                                        'concat("Parámetros enviados ahora mismo -> assetId=\'", {view.custom.selectedAsset}, '
                                                        '"\', simCycle=\'", {view.custom.simCycle}, "\' | Fila devuelta en view.custom.kpis[0]:")'
                                                    )
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "style": {
                                            "color": "#CBD5E1",
                                            "fontSize": "11px",
                                            "fontFamily": "JetBrains Mono, Consolas, monospace",
                                        }
                                    },
                                },
                                {
                                    "meta": {"name": "Table_Raw_KPI_Query"},
                                    "type": "ia.display.table",
                                    "position": {"basis": "115px", "shrink": 0},
                                    "propConfig": {
                                        "props.data": {
                                            "binding": {
                                                "type": "property",
                                                "config": {
                                                    "path": "view.custom.kpis",
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "pager": {"bottom": False},
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "color": "#38BDF8",
                                            "borderRadius": "8px",
                                            "fontSize": "11px",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Lbl_Chart_Guia_Title"},
                                    "type": "ia.display.label",
                                    "position": {"basis": "22px", "shrink": 0},
                                    "props": {
                                        "text": "📈 GRÁFICA EN VIVO DEL NAMED QUERY (CMAPSS/telemetry_trend HASTA :simCycle)",
                                        "style": {
                                            "color": "#10B981",
                                            "fontSize": "12px",
                                            "fontWeight": "800",
                                            "letterSpacing": "0.03em",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "TimeSeriesChart_Guia"},
                                    "type": "ia.chart.timeseries",
                                    "position": {"basis": "240px", "grow": 1},
                                    "propConfig": {
                                        "props.series[0].data": {
                                            "binding": {
                                                "type": "query",
                                                "config": {
                                                    "parameters": {
                                                        "assetId": "{view.custom.selectedAsset}",
                                                        "startCycle": "1",
                                                        "endCycle": "{view.custom.simCycle}",
                                                    },
                                                    "queryPath": "CMAPSS/telemetry_trend",
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "series": [{"name": "DegradacionTermica"}],
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "borderRadius": "8px",
                                            "padding": "6px",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Table_Sensors_Guia"},
                                    "type": "ia.display.table",
                                    "position": {"basis": "185px", "grow": 1},
                                    "propConfig": {
                                        "props.data": {
                                            "binding": {
                                                "type": "query",
                                                "config": {
                                                    "parameters": {
                                                        "assetId": "{view.custom.selectedAsset}"
                                                    },
                                                    "queryPath": "CMAPSS/latest_sensors",
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "pager": {"bottom": False},
                                        "rows": {"striped": True},
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "color": "#E2E8F0",
                                            "borderRadius": "8px",
                                            "fontSize": "11px",
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    }


def deploy_perspective_hmi(
    container_name: str = "twin-ignition",
    project: str = "DigitalTwin",
    view_name: str = "Turbofan_Overview",
) -> dict[str, Any]:
    """Despliega la vista operativa (`Turbofan_Overview` en `/`) y la vista educativa (`Arquitectura_Guia` en `/guia`)."""
    view_payload = generate_isa101_perspective_view()
    SCADA_DIR.mkdir(parents=True, exist_ok=True)
    local_view_path = SCADA_DIR / "view.json"
    with open(local_view_path, "w", encoding="utf-8") as f:
        json.dump(view_payload, f, ensure_ascii=False, indent=2)

    # Generar también la nueva vista educativa Arquitectura_Guia (/guia)
    guia_payload = generate_architecture_guide_view()
    local_guia_view_path = SCADA_DIR / "guia_view.json"
    with open(local_guia_view_path, "w", encoding="utf-8") as f:
        json.dump(guia_payload, f, ensure_ascii=False, indent=2)

    guia_resource = {
        "scope": "G",
        "version": 1,
        "restricted": False,
        "overridable": True,
        "files": ["view.json"],
        "attributes": {
            "lastModificationSignature": "23e315a966e13a086a58eca446614337d096b9523717aa33a509cc1e16309a8c",
            "lastModification": {
                "actor": "admin",
                "timestamp": "2026-10-26T23:55:00Z",
            },
        },
    }
    local_guia_res_path = SCADA_DIR / "guia_resource.json"
    with open(local_guia_res_path, "w", encoding="utf-8") as f:
        json.dump(guia_resource, f, indent=2)

    page_config = {
        "pages": {
            "/": {"viewPath": "Turbofan_Overview"},
            "/guia": {"viewPath": "Arquitectura_Guia"},
        },
        "sharedDocks": {"cornerPriority": "top-bottom"},
    }
    local_page_cfg_path = SCADA_DIR / "page_config.json"
    with open(local_page_cfg_path, "w", encoding="utf-8") as f:
        json.dump(page_config, f, indent=2)

    # SQL de turbofan_kpis parametrizado por :simCycle para simular la degradación ciclo a ciclo
    upgraded_kpi_sql = """WITH latest_seq AS (
    SELECT 
        COALESCE(
            MAX(CASE WHEN sequence_no <= :simCycle THEN sequence_no END),
            MAX(sequence_no)
        ) AS eval_seq,
        MAX(sequence_no) AS max_seq
    FROM telemetry
    WHERE asset_id = :assetId
),
latest_pred AS (
    SELECT 
        estimate AS end_rul,
        lower_bound AS end_lower_90,
        upper_bound AS end_upper_90
    FROM prediction
    WHERE asset_id = :assetId
    ORDER BY sequence_no DESC, prediction_id DESC
    LIMIT 1
)
SELECT 
    t.asset_id,
    t.sequence_no AS current_cycle,
    MAX(t.observed_at) AS last_observed_at,
    MAX(CASE WHEN t.signal_key = 's1' THEN t.value_double END) AS fan_inlet_temp_s1,
    MAX(CASE WHEN t.signal_key = 's2' THEN t.value_double END) AS lpc_temp_s2,
    MAX(CASE WHEN t.signal_key = 's3' THEN t.value_double END) AS hpc_temp_s3,
    MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) AS lpt_temp_s4,
    MAX(CASE WHEN t.signal_key = 's7' THEN t.value_double END) AS hpc_press_s7,
    MAX(CASE WHEN t.signal_key = 's8' THEN t.value_double END) AS fan_speed_s8,
    MAX(CASE WHEN t.signal_key = 's9' THEN t.value_double END) AS core_speed_s9,
    MAX(CASE WHEN t.signal_key = 's11' THEN t.value_double END) AS hpc_static_press_s11,
    MAX(CASE WHEN t.signal_key = 's12' THEN t.value_double END) AS fuel_flow_s12,
    MAX(CASE WHEN t.signal_key = 's21' THEN t.value_double END) AS lpt_bleed_s21,
    CASE 
        WHEN MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) >= 1418.0 THEN 'Critico'
        WHEN MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) >= 1408.0 THEN 'Alerta'
        ELSE 'Normal'
    END AS health_status,
    LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) AS rul_cycles,
    LEAST(125.0, GREATEST(0.0, COALESCE(MAX(lp.end_lower_90), 0.0) + (MAX(ls.max_seq) - t.sequence_no))) AS rul_lower_90,
    LEAST(125.0, COALESCE(MAX(lp.end_upper_90), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) AS rul_upper_90,
    CASE 
        WHEN LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) <= 15.0 THEN 'Parada Inmediata'
        WHEN LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) <= 45.0 THEN 'Mantenimiento Requerido'
        ELSE 'Operación Segura'
    END AS prescriptive_action
FROM telemetry t
JOIN latest_seq ls ON t.sequence_no = ls.eval_seq
LEFT JOIN latest_pred lp ON TRUE
WHERE t.asset_id = :assetId
GROUP BY t.asset_id, t.sequence_no;
"""
    local_sql_path = SCADA_DIR / "turbofan_kpis.sql"
    with open(local_sql_path, "w", encoding="utf-8") as f:
        f.write(upgraded_kpi_sql)

    # Actualizamos también resource.json de turbofan_kpis para declarar el parámetro entero :simCycle (sqlType: 2)
    kpi_resource = {
        "scope": "DG",
        "version": 2,
        "restricted": False,
        "overridable": True,
        "files": ["query.sql"],
        "attributes": {
            "useMaxReturnSize": False,
            "autoBatchEnabled": False,
            "fallbackValue": "",
            "maxReturnSize": 100,
            "cacheUnit": "SEC",
            "type": "Query",
            "enabled": True,
            "cacheAmount": 1,
            "cacheEnabled": False,
            "database": "twin",
            "fallbackEnabled": False,
            "lastModificationSignature": "390c212f90e695183c448ba83c9733ac2c017587b3f18fae4471ad66047354c7",
            "permissions": [{"zone": "", "role": ""}],
            "lastModification": {"actor": "admin", "timestamp": "2026-10-26T23:45:00Z"},
            "parameters": [
                {"type": "Parameter", "identifier": "assetId", "sqlType": 7},
                {"type": "Parameter", "identifier": "simCycle", "sqlType": 2},
            ],
        },
    }
    local_kpi_res_path = SCADA_DIR / "turbofan_kpis_resource.json"
    with open(local_kpi_res_path, "w", encoding="utf-8") as f:
        json.dump(kpi_resource, f, indent=2)

    guia_dir_in_container = (
        f"/usr/local/bin/ignition/data/projects/{project}/"
        "com.inductiveautomation.perspective/views/Arquitectura_Guia"
    )
    subprocess.run(
        ["docker", "exec", container_name, "mkdir", "-p", guia_dir_in_container],
        capture_output=True,
        text=True,
    )

    container_view_path = (
        f"{container_name}:/usr/local/bin/ignition/data/projects/{project}/"
        f"com.inductiveautomation.perspective/views/{view_name}/view.json"
    )
    container_guia_view_path = f"{container_name}:{guia_dir_in_container}/view.json"
    container_guia_res_path = f"{container_name}:{guia_dir_in_container}/resource.json"
    container_page_cfg_path = (
        f"{container_name}:/usr/local/bin/ignition/data/projects/{project}/"
        "com.inductiveautomation.perspective/page-config/config.json"
    )
    container_sql_path = (
        f"{container_name}:/usr/local/bin/ignition/data/projects/{project}/"
        f"ignition/named-query/CMAPSS/turbofan_kpis/query.sql"
    )
    container_kpi_res_path = (
        f"{container_name}:/usr/local/bin/ignition/data/projects/{project}/"
        f"ignition/named-query/CMAPSS/turbofan_kpis/resource.json"
    )

    for src_p, dst_p in [
        (local_view_path, container_view_path),
        (local_guia_view_path, container_guia_view_path),
        (local_guia_res_path, container_guia_res_path),
        (local_page_cfg_path, container_page_cfg_path),
        (local_sql_path, container_sql_path),
        (local_kpi_res_path, container_kpi_res_path),
    ]:
        cp_res = subprocess.run(["docker", "cp", str(src_p), dst_p], capture_output=True, text=True)
        if cp_res.returncode != 0:
            raise RuntimeError(f"Error al copiar {src_p.name} al contenedor: {cp_res.stderr.strip()}")

    return {
        "status": "DEPLOYED",
        "container": container_name,
        "project": project,
        "view_name": view_name,
        "guide_view_name": "Arquitectura_Guia",
        "perspective_url": f"http://localhost:8089/data/perspective/client/{project}",
        "guide_url": f"http://localhost:8089/data/perspective/client/{project}/guia",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }


def _create_fastmcp_server():
    """Inicializa el servidor FastMCP si el SDK oficial `mcp` está instalado."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("cmapss-digital-twin")

    @mcp.tool(name="get_fleet_status")
    def tool_get_fleet_status(subset: str = "FD001", split: str = "test", top_n: int = 10) -> dict[str, Any]:
        """Devuelve el resumen de salud de la flota de turbinas y los motores con menor vida útil restante (RUL)."""
        return get_fleet_status(subset=subset, split=split, top_n=top_n)

    @mcp.tool(name="predict_engine_rul")
    def tool_predict_engine_rul(
        asset_id: str,
        cycle: int | None = None,
        subset: str = "FD001",
        split: str = "test",
    ) -> dict[str, Any]:
        """Predice el RUL (en ciclos) de una turbina específica y muestra sus sensores más influyentes."""
        return predict_engine_rul(asset_id=asset_id, cycle=cycle, subset=subset, split=split)

    @mcp.tool(name="get_sensor_telemetry")
    def tool_get_sensor_telemetry(
        asset_id: str,
        last_n_cycles: int = 10,
        subset: str = "FD001",
        split: str = "test",
    ) -> dict[str, Any]:
        """Obtiene las últimas lecturas de los sensores de una turbina y calcula cuáles presentan mayor deriva física."""
        return get_sensor_telemetry(
            asset_id=asset_id,
            last_n_cycles=last_n_cycles,
            subset=subset,
            split=split,
        )

    @mcp.tool(name="get_model_diagnostics")
    def tool_get_model_diagnostics(subset: str = "FD001") -> dict[str, Any]:
        """Devuelve las métricas de evaluación del modelo XGBoost (RMSE, MAE, R², NASA S-Score) e importancia de variables."""
        return get_model_diagnostics(subset=subset)

    @mcp.tool(name="audit_perspective_hmi")
    def tool_audit_perspective_hmi(
        container_name: str = "twin-ignition",
        project: str = "DigitalTwin",
        view_name: str = "Turbofan_Overview",
    ) -> dict[str, Any]:
        """Inspecciona la vista actual de Ignition Perspective en Docker y detecta errores de bindings y brechas ISA-101."""
        return audit_perspective_hmi(container_name=container_name, project=project, view_name=view_name)

    @mcp.tool(name="deploy_perspective_hmi")
    def tool_deploy_perspective_hmi(
        container_name: str = "twin-ignition",
        project: str = "DigitalTwin",
        view_name: str = "Turbofan_Overview",
    ) -> dict[str, Any]:
        """Genera y despliega directamente en el contenedor Ignition la vista HMI con Dibujo Animado de Turbina y Simulador."""
        return deploy_perspective_hmi(container_name=container_name, project=project, view_name=view_name)

    @mcp.resource("cmapss://manifest")
    def resource_manifest() -> str:
        """Manifiesto del dataset C-MAPSS con descripción física y unidades de los 21 sensores."""
        return json.dumps(_get_manifest(), ensure_ascii=False, indent=2)

    @mcp.resource("cmapss://fleet/critical")
    def resource_critical_fleet() -> str:
        """Snapshot JSON de los motores más críticos en el conjunto de prueba FD001."""
        return json.dumps(get_fleet_status(subset="FD001", split="test", top_n=10), ensure_ascii=False, indent=2)

    return mcp


def run_demo() -> None:
    """Despliega y audita la vista HMI con el Dibujo Animado SVG del Turbofan y el Simulador."""
    deploy_res = deploy_perspective_hmi()
    audit_res = audit_perspective_hmi()
    print(json.dumps({"deploy": deploy_res, "audit": audit_res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        server = _create_fastmcp_server()
        server.run()
