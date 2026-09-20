#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华山项目纵向生化数据清洗脚本
================================
目标：
1. 从原始宽表 .xlsx 中提取 T0 基线信息与 T1~Tn 随访信息；
2. 核心预测指标：
   HBA1C, TG, CHO, HDL, LDL, ALB, BUN, UA, CRE
3. CysC 作为第 10 个额外指标：
   - 保留原始缺失值 NaN
   - 生成 missing mask：1=有观测，0=缺失
4. 将宽表整理成长表，适用于：
   CT + T0生化 + demographics + future time -> future biochemical prediction
5. 不在此阶段做标准化、归一化或中位数填补：
   这些操作必须在 train/val/test 划分以后，仅用训练集拟合，避免数据泄漏。

默认读取工作表：
    小于等于一个月
即主分析 cohort（CT 与 T0 生化间隔 <= 30 天）。

如需做 <= 60 天敏感性分析，可同时读取：
    小于等于一个月 小于等于两个月

依赖：
    pip install pandas openpyxl

示例：
    python clean_huashan_dataset.py --input try.xlsx --output huashan_cleaned.xlsx

    python clean_huashan_dataset.py \
        --input try.xlsx \
        --output huashan_cleaned_60d.xlsx \
        --sheets 小于等于一个月 小于等于两个月
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORE_BIOMARKERS = [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
]
EXTRA_BIOMARKER = "CysC"

DEFAULT_SHEETS = ["小于等于一个月"]

BASE_PREFIX = "T0_清洗结果｜"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗华山项目 CT + 纵向生化 Excel，并转换为长表格式。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入 .xlsx 文件，例如 try.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("huashan_cleaned_7000.xlsx"),
        help="输出 .xlsx 文件，默认 huashan_cleaned.xlsx",
    )
    parser.add_argument(
        "--sheets",
        nargs="+",
        default=DEFAULT_SHEETS,
        help=(
            "需要合并清洗的工作表。默认：小于等于一个月。"
            "例如 <=60天可传入：小于等于一个月 小于等于两个月"
        ),
    )
    parser.add_argument(
        "--keep-core-missing",
        action="store_true",
        help=(
            "默认会删除 T0 核心9项缺失的患者，以及核心9项目标缺失的随访记录。"
            "指定此参数后保留它们（通常不建议用于第一版模型）。"
        ),
    )
    return parser.parse_args()


def clean_patient_id(value) -> str | pd._libs.missing.NAType:
    """将 Excel 中可能被读成 int/float 的 PatientIntraID 统一转成字符串。"""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def numeric_series(series: pd.Series) -> pd.Series:
    """稳健地将生化/时间差列转成数值；异常字符串转成 NaN。"""
    if series is None:
        return pd.Series(dtype="float64")

    if series.dtype == object:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA})
        )
        return pd.to_numeric(cleaned, errors="coerce")

    return pd.to_numeric(series, errors="coerce")


def parse_age(series: pd.Series) -> pd.Series:
    """支持 '89岁'、'89'、89 等形式。"""
    extracted = (
        series.astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def parse_excel_date(series: pd.Series) -> pd.Series:
    """
    同时兼容：
    - Excel serial date，例如 43130
    - pandas 已识别的 Timestamp/datetime
    - '2018-01-30' 等字符串
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    # 先识别典型 Excel serial date，避免 pd.to_datetime(43130)
    # 被误解释成 1970 年附近的纳秒时间戳。
    numeric = pd.to_numeric(series, errors="coerce")
    excel_serial_mask = numeric.between(20000, 80000, inclusive="both")

    if excel_serial_mask.any():
        result.loc[excel_serial_mask] = (
            pd.Timestamp("1899-12-30")
            + pd.to_timedelta(numeric.loc[excel_serial_mask], unit="D")
        )

    remaining = ~excel_serial_mask & series.notna()
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(
            series.loc[remaining],
            errors="coerce",
        )

    return result


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context} 缺少必要列：\n  - " + "\n  - ".join(missing)
        )


def discover_followup_indices(columns: Iterable[str]) -> list[int]:
    """自动发现 T1、T2 ... T24，不硬编码最大随访次数。"""
    indices = []
    pattern = re.compile(r"^T(\d+)_距T0天数$")
    for col in columns:
        match = pattern.match(str(col))
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def build_baseline(df: pd.DataFrame, source_sheet: str) -> pd.DataFrame:
    required = [
        "PatientIntraID",
        "Comparison status",
        "T0_最终匹配的CT序列具体地址",
        "T0_CT拍摄时间",
        "T0_时间差（天）",
        f"{BASE_PREFIX}性别",
        f"{BASE_PREFIX}年龄",
        f"{BASE_PREFIX}样本日期",
    ]
    required += [f"{BASE_PREFIX}{b}" for b in CORE_BIOMARKERS]
    required += [f"{BASE_PREFIX}{EXTRA_BIOMARKER}"]

    require_columns(df, required, f"工作表 {source_sheet} / baseline")

    out = pd.DataFrame(index=df.index)

    # ---------- patient / CT / demographics ----------
    out["patient_id"] = df["PatientIntraID"].map(clean_patient_id)
    out["ct_path"] = df["T0_最终匹配的CT序列具体地址"].astype("string")
    out["comparison_status"] = df["Comparison status"].astype("string")
    out["sex"] = df[f"{BASE_PREFIX}性别"].astype("string").str.strip()
    out["sex_code"] = out["sex"].map({"女": 0, "男": 1}).astype("Int64")
    out["age"] = parse_age(df[f"{BASE_PREFIX}年龄"])

    out["ct_date"] = parse_excel_date(df["T0_CT拍摄时间"])
    out["baseline_date"] = parse_excel_date(df[f"{BASE_PREFIX}样本日期"])
    out["ct_bio_gap_days"] = numeric_series(df["T0_时间差（天）"])

    # ---------- T0 biochemical input ----------
    for biomarker in CORE_BIOMARKERS:
        out[f"base_{biomarker}"] = numeric_series(
            df[f"{BASE_PREFIX}{biomarker}"]
        )

    out[f"base_{EXTRA_BIOMARKER}"] = numeric_series(
        df[f"{BASE_PREFIX}{EXTRA_BIOMARKER}"]
    )

    # mask 约定：1 = observed，0 = missing
    out[f"base_{EXTRA_BIOMARKER}_mask"] = (
        out[f"base_{EXTRA_BIOMARKER}"].notna().astype("int8")
    )

    out["baseline_source_sheet"] = source_sheet

    return out


def build_followup_long(df: pd.DataFrame, source_sheet: str) -> tuple[pd.DataFrame, dict]:
    followup_indices = discover_followup_indices(df.columns)
    if not followup_indices:
        raise ValueError(f"工作表 {source_sheet} 中没有找到 T1/T2/... 随访列。")

    pieces = []
    qc = {
        "followup_indices_found": len(followup_indices),
        "raw_followup_rows": 0,
        "invalid_or_nonfuture_time_rows": 0,
    }

    for t in followup_indices:
        required = [
            "PatientIntraID",
            f"T{t}_距T0天数",
            f"T{t}_样本日期",
        ]
        required += [f"T{t}_{b}" for b in CORE_BIOMARKERS]
        required += [f"T{t}_{EXTRA_BIOMARKER}"]
        require_columns(df, required, f"工作表 {source_sheet} / T{t}")

        t_days = numeric_series(df[f"T{t}_距T0天数"])
        has_followup = t_days.notna()

        if not has_followup.any():
            continue

        tmp = pd.DataFrame(index=df.index[has_followup])
        tmp["patient_id"] = df.loc[has_followup, "PatientIntraID"].map(clean_patient_id)
        tmp["followup_index"] = t
        tmp["t_days"] = t_days.loc[has_followup]
        tmp["followup_date"] = parse_excel_date(
            df.loc[has_followup, f"T{t}_样本日期"]
        )

        for biomarker in CORE_BIOMARKERS:
            tmp[f"target_{biomarker}"] = numeric_series(
                df.loc[has_followup, f"T{t}_{biomarker}"]
            )

        tmp[f"target_{EXTRA_BIOMARKER}"] = numeric_series(
            df.loc[has_followup, f"T{t}_{EXTRA_BIOMARKER}"]
        )
        tmp[f"target_{EXTRA_BIOMARKER}_mask"] = (
            tmp[f"target_{EXTRA_BIOMARKER}"].notna().astype("int8")
        )

        tmp["followup_source_sheet"] = source_sheet

        qc["raw_followup_rows"] += len(tmp)

        # 预测任务只保留真正未来的时间点
        invalid_time = tmp["t_days"].isna() | (tmp["t_days"] <= 0)
        qc["invalid_or_nonfuture_time_rows"] += int(invalid_time.sum())
        tmp = tmp.loc[~invalid_time].copy()

        pieces.append(tmp)

    if not pieces:
        return pd.DataFrame(), qc

    followup = pd.concat(pieces, ignore_index=True)

    # 同一患者若存在完全相同的随访时间，保留原记录但标记，便于后续人工 QC
    followup["same_patient_same_day_duplicate"] = (
        followup.duplicated(["patient_id", "t_days"], keep=False).astype("int8")
    )

    # 时间窗口标签：暂不删除 >2 年数据，避免在清洗阶段不可逆丢失信息
    followup["time_window"] = pd.cut(
        followup["t_days"],
        bins=[0, 365, 730, np.inf],
        labels=["0-1y", "1-2y", ">2y"],
        right=True,
    ).astype("string")

    followup = followup.sort_values(
        ["patient_id", "t_days", "followup_index"],
        kind="stable",
    ).reset_index(drop=True)

    return followup, qc


def clean_one_sheet(
    df: pd.DataFrame,
    source_sheet: str,
    keep_core_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = build_baseline(df, source_sheet)
    followup, followup_qc = build_followup_long(df, source_sheet)

    qc = {
        "source_sheet": source_sheet,
        "raw_patients": len(df),
        "duplicate_patient_id_rows": int(
            baseline["patient_id"].duplicated(keep=False).sum()
        ),
        **followup_qc,
    }

    # 当前 workbook 的每个 cohort sheet 理论上应为“一人一行”
    if baseline["patient_id"].isna().any():
        raise ValueError(f"工作表 {source_sheet} 中存在空 PatientIntraID。")

    if baseline["patient_id"].duplicated().any():
        dup_ids = (
            baseline.loc[
                baseline["patient_id"].duplicated(keep=False),
                "patient_id",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"工作表 {source_sheet} 中 PatientIntraID 重复，"
            f"请先核对：{dup_ids[:20]}"
        )

    base_core_cols = [f"base_{b}" for b in CORE_BIOMARKERS]
    target_core_cols = [f"target_{b}" for b in CORE_BIOMARKERS]

    baseline_missing_core = baseline[base_core_cols].isna().any(axis=1)
    followup_missing_core = (
        followup[target_core_cols].isna().any(axis=1)
        if not followup.empty
        else pd.Series(dtype=bool)
    )

    qc["baseline_rows_missing_any_core"] = int(baseline_missing_core.sum())
    qc["followup_rows_missing_any_core"] = int(followup_missing_core.sum())

    if not keep_core_missing:
        valid_patient_ids = set(
            baseline.loc[~baseline_missing_core, "patient_id"].dropna()
        )
        baseline = baseline.loc[
            baseline["patient_id"].isin(valid_patient_ids)
        ].copy()

        if not followup.empty:
            followup = followup.loc[
                followup["patient_id"].isin(valid_patient_ids)
                & ~followup_missing_core
            ].copy()

    qc["final_baseline_patients"] = len(baseline)
    qc["final_followup_rows"] = len(followup)

    return baseline, followup, qc


def make_qc_summary(
    baseline: pd.DataFrame,
    followup: pd.DataFrame,
    per_sheet_qc: list[dict],
) -> pd.DataFrame:
    rows = []

    def add(metric: str, value, note: str = ""):
        rows.append({"metric": metric, "value": value, "note": note})

    add("final_patient_count", int(baseline["patient_id"].nunique()))
    add("final_followup_observation_count", int(len(followup)))

    if not followup.empty:
        add("followup_0_1y_rows", int((followup["time_window"] == "0-1y").sum()))
        add("followup_1_2y_rows", int((followup["time_window"] == "1-2y").sum()))
        add("followup_gt_2y_rows", int((followup["time_window"] == ">2y").sum()))

        add(
            "patients_with_any_0_1y_followup",
            int(followup.loc[followup["time_window"] == "0-1y", "patient_id"].nunique()),
        )
        add(
            "patients_with_any_1_2y_followup",
            int(followup.loc[followup["time_window"] == "1-2y", "patient_id"].nunique()),
        )

    base_mask_col = f"base_{EXTRA_BIOMARKER}_mask"
    target_mask_col = f"target_{EXTRA_BIOMARKER}_mask"

    add(
        "baseline_CysC_observed",
        int(baseline[base_mask_col].sum()),
        "mask=1",
    )
    add(
        "baseline_CysC_missing",
        int((baseline[base_mask_col] == 0).sum()),
        "mask=0",
    )

    if not followup.empty:
        add(
            "target_CysC_observed",
            int(followup[target_mask_col].sum()),
            "mask=1",
        )
        add(
            "target_CysC_missing",
            int((followup[target_mask_col] == 0).sum()),
            "mask=0",
        )
        add(
            "same_patient_same_day_duplicate_rows",
            int(followup["same_patient_same_day_duplicate"].sum()),
            "建议人工检查，不在清洗阶段自动删除",
        )

    for q in per_sheet_qc:
        sheet = q["source_sheet"]
        add(f"[{sheet}] raw_patients", q["raw_patients"])
        add(f"[{sheet}] raw_followup_rows", q["raw_followup_rows"])
        add(
            f"[{sheet}] baseline_missing_any_core",
            q["baseline_rows_missing_any_core"],
        )
        add(
            f"[{sheet}] followup_missing_any_core",
            q["followup_rows_missing_any_core"],
        )
        add(
            f"[{sheet}] invalid_or_nonfuture_time_rows",
            q["invalid_or_nonfuture_time_rows"],
        )

    add(
        "CysC_mask_definition",
        "1=observed; 0=missing",
        "CysC 缺失不删除；训练时使用 mask 控制输入/损失。",
    )
    add(
        "normalization_note",
        "NOT PERFORMED",
        "标准化/归一化必须在 patient-level 数据划分后，仅使用训练集拟合。",
    )
    add(
        "CysC_imputation_note",
        "NOT PERFORMED",
        "CysC 原始 NaN 保留；后续仅用训练集统计量填补，并同时输入 missing mask。",
    )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件：{args.input}")

    xls = pd.ExcelFile(args.input, engine="openpyxl")
    available_sheets = xls.sheet_names

    missing_sheets = [s for s in args.sheets if s not in available_sheets]
    if missing_sheets:
        raise ValueError(
            "以下工作表不存在："
            + ", ".join(missing_sheets)
            + "\n当前文件包含："
            + ", ".join(available_sheets)
        )

    baseline_parts = []
    followup_parts = []
    per_sheet_qc = []

    for sheet_name in args.sheets:
        print(f"[INFO] 正在读取工作表：{sheet_name}")
        df = pd.read_excel(
            args.input,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        # 去掉完全空的行/列；不修改原始文件
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        baseline, followup, qc = clean_one_sheet(
            df=df,
            source_sheet=sheet_name,
            keep_core_missing=args.keep_core_missing,
        )

        baseline_parts.append(baseline)
        followup_parts.append(followup)
        per_sheet_qc.append(qc)

    baseline_all = pd.concat(baseline_parts, ignore_index=True)
    followup_all = pd.concat(followup_parts, ignore_index=True)

    # 多 sheet 合并后再次确认同一患者没有跨 cohort 重复
    duplicated = baseline_all["patient_id"].duplicated(keep=False)
    if duplicated.any():
        dup = (
            baseline_all.loc[duplicated, ["patient_id", "baseline_source_sheet"]]
            .sort_values("patient_id")
        )
        raise ValueError(
            "多个工作表合并后发现 PatientIntraID 重复。"
            "请确认这些 cohort 是否互斥。\n"
            + dup.head(30).to_string(index=False)
        )

    # 仅保留 baseline 中存在的患者
    valid_ids = set(baseline_all["patient_id"])
    followup_all = followup_all[
        followup_all["patient_id"].isin(valid_ids)
    ].copy()

    # 模型长表：每个随访 observation 一行，重复携带该患者的 T0 输入
    model_long = followup_all.merge(
        baseline_all,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    # 更符合当前研究目标的 0~2 年 ready-to-train 子集。
    # 仍保留 model_long_all，故不会丢失 >2 年的原始清洗结果。
    model_long_0_2y = model_long[
        (model_long["t_days"] > 0)
        & (model_long["t_days"] <= 730)
    ].copy()

    qc_summary = make_qc_summary(
        baseline=baseline_all,
        followup=followup_all,
        per_sheet_qc=per_sheet_qc,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        args.output,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        baseline_all.to_excel(writer, sheet_name="baseline", index=False)
        followup_all.to_excel(writer, sheet_name="longitudinal_all", index=False)
        model_long.to_excel(writer, sheet_name="model_long_all", index=False)
        model_long_0_2y.to_excel(writer, sheet_name="model_long_0_2y", index=False)
        qc_summary.to_excel(writer, sheet_name="qc_summary", index=False)

    print("\n[DONE] 清洗完成")
    print(f"输出文件：{args.output.resolve()}")
    print(f"患者数：{baseline_all['patient_id'].nunique()}")
    print(f"全部有效随访 observation：{len(followup_all)}")
    print(f"0~2 年 observation：{len(model_long_0_2y)}")
    print(
        "Baseline CysC 缺失："
        f"{int((baseline_all[f'base_{EXTRA_BIOMARKER}_mask'] == 0).sum())}"
    )
    print(
        "Target CysC 缺失："
        f"{int((followup_all[f'target_{EXTRA_BIOMARKER}_mask'] == 0).sum())}"
    )
    print("\n注意：本脚本没有做标准化，也没有填补 CysC。")
    print("这些步骤应在 patient-level train/val/test 划分后，仅用训练集拟合。")


if __name__ == "__main__":
    main()
