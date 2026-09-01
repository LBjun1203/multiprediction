#!/usr/bin/env python3
"""更新 huashan_cleaned.xlsx 中 baseline 工作表的 CT 路径。

默认将每一行的 ``ct_path`` 改为：

    E:\\data\\huashan_349\\<patient_id>

脚本默认原地更新工作簿，也可以通过 ``--output`` 另存为新文件。

依赖：
    python -m pip install openpyxl
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    from openpyxl import load_workbook
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 openpyxl，请先运行：python -m pip install openpyxl"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = SCRIPT_DIR / "huashan_cleaned.xlsx"
DEFAULT_TARGET_ROOT = PureWindowsPath(r"E:\data\huashan_349")
SHEET_NAME = "baseline"
PATIENT_ID_COLUMN = "patient_id"
CT_PATH_COLUMN = "ct_path"


def normalize_patient_id(value: Any) -> str:
    """将 Excel 中的患者编号转换为适合拼接路径的字符串。"""

    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"患者编号不是整数：{value!r}")
        return str(int(value))
    return str(value).strip()


def find_columns(worksheet) -> tuple[int, int, int]:
    """查找表头行以及 patient_id、ct_path 两列的位置。"""

    for header_row in range(1, min(worksheet.max_row, 50) + 1):
        columns: dict[str, int] = {}
        for cell in worksheet[header_row]:
            if cell.value is None:
                continue
            header = str(cell.value).strip()
            if header in {PATIENT_ID_COLUMN, CT_PATH_COLUMN}:
                if header in columns:
                    raise ValueError(
                        f"工作表 {SHEET_NAME!r} 第 {header_row} 行存在重复列：{header!r}"
                    )
                columns[header] = cell.column

        if PATIENT_ID_COLUMN in columns and CT_PATH_COLUMN in columns:
            return (
                header_row,
                columns[PATIENT_ID_COLUMN],
                columns[CT_PATH_COLUMN],
            )

    raise ValueError(
        f"工作表 {SHEET_NAME!r} 的前 50 行中未找到 "
        f"{PATIENT_ID_COLUMN!r} 和 {CT_PATH_COLUMN!r} 表头"
    )


def update_ct_paths(
    input_path: Path,
    output_path: Path,
    target_root: PureWindowsPath,
) -> int:
    """更新 baseline.ct_path，并返回被更新的行数。"""

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入工作簿：{input_path}")

    workbook = load_workbook(input_path)
    try:
        if SHEET_NAME not in workbook.sheetnames:
            raise ValueError(f"工作簿中不存在工作表：{SHEET_NAME!r}")

        worksheet = workbook[SHEET_NAME]
        header_row, patient_id_column, ct_path_column = find_columns(worksheet)

        updated = 0
        for row_number in range(header_row + 1, worksheet.max_row + 1):
            patient_id = normalize_patient_id(
                worksheet.cell(row=row_number, column=patient_id_column).value
            )
            if not patient_id:
                continue

            worksheet.cell(row=row_number, column=ct_path_column).value = str(
                target_root / patient_id
            )
            updated += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}.",
                suffix=output_path.suffix,
                dir=output_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            workbook.save(temporary_path)
            os.replace(temporary_path, output_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return updated
    finally:
        workbook.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 huashan_cleaned.xlsx 的 baseline.ct_path 更新为 "
            r"E:\data\huashan_349\<patient_id>。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="输入工作簿；默认读取脚本同目录下的 huashan_cleaned.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出工作簿；默认原地更新输入工作簿",
    )
    parser.add_argument(
        "--target-root",
        default=str(DEFAULT_TARGET_ROOT),
        help=r"CT 根目录；默认为 E:\data\huashan_349",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path
    updated = update_ct_paths(
        input_path=input_path,
        output_path=output_path,
        target_root=PureWindowsPath(args.target_root),
    )
    print(f"已更新 {updated} 行 ct_path：{output_path}")


if __name__ == "__main__":
    main()
