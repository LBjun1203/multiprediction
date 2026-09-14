"""统计 Excel 中 CT 地址对应的 dicom 文件夹大小。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook


DEFAULT_EXCEL = (
    Path(__file__).resolve().parent
    / "data_washing"
    / "archive"
    / "all_results"
    / "all_results_final_deduplicated_with_ct_time.xlsx"
)
DEFAULT_SHEET = "Results"
DEFAULT_COLUMN = "最终匹配的CT序列具体地址"


def human_size(size: int) -> str:
    """将字节数格式化为易读的 IEC 单位。"""
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def normalize_path(value: object) -> Path | None:
    """清理 Excel 单元格中的路径文本。"""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None
    return Path(os.path.expandvars(os.path.expanduser(text)))


def find_dicom_dirs(address: Path) -> list[Path]:
    """找到地址本身代表的 DICOM 序列，或其下名为 dicom 的目录。"""
    if not address.is_dir():
        return []
    if address.name.casefold() == "dicom":
        return [address]

    # 这份工作簿中的地址通常就是序列目录，DICOM 文件直接存放于其中。
    try:
        with os.scandir(address) as entries:
            if any(
                entry.is_file(follow_symlinks=False)
                and Path(entry.name).suffix.casefold() in {".dcm", ".dicom"}
                for entry in entries
            ):
                return [address]
    except OSError:
        return []

    found: list[Path] = []
    # 找到 dicom 后不再进入该目录，避免重复扫描其中可能存在的同名子目录。
    for root, dirnames, _ in os.walk(address, followlinks=False):
        matched = [name for name in dirnames if name.casefold() == "dicom"]
        for name in matched:
            found.append(Path(root) / name)
        dirnames[:] = [name for name in dirnames if name.casefold() != "dicom"]
    return found


def directory_size(directory: Path) -> tuple[int, int, list[str]]:
    """返回目录总字节数、文件数以及无法读取的项目。"""
    total = 0
    file_count = 0
    errors: list[str] = []

    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            entries = os.scandir(current)
        except OSError as exc:
            errors.append(f"{current}: {exc}")
            continue
        with entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        file_count += 1
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
    return total, file_count, errors


def canonical_key(path: Path) -> str:
    """生成适合 Windows 路径去重的键。"""
    return os.path.normcase(os.path.abspath(path))


def iter_addresses(
    excel_path: Path, sheet_name: str, column_name: str
) -> Iterable[tuple[int, object]]:
    workbook = load_workbook(excel_path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(
                f"找不到工作表 {sheet_name!r}；现有工作表：{workbook.sheetnames}"
            )
        sheet = workbook[sheet_name]
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        normalized = [str(value).strip() if value is not None else "" for value in header]
        if column_name not in normalized:
            raise ValueError(
                f"找不到列 {column_name!r}；现有列：{normalized}"
            )
        column_index = normalized.index(column_name)
        for row_number, row in enumerate(
            sheet.iter_rows(min_row=2, values_only=True), start=2
        ):
            yield row_number, row[column_index]
    finally:
        workbook.close()


def write_details(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("excel_row", "address", "dicom_dirs", "bytes", "files", "status")
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 Results 工作表中 CT 地址下 dicom 文件夹的数据量。"
    )
    parser.add_argument(
        "excel",
        nargs="?",
        type=Path,
        default=DEFAULT_EXCEL,
        help=f"输入工作簿（默认：{DEFAULT_EXCEL}）",
    )
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="工作表名称")
    parser.add_argument("--column", default=DEFAULT_COLUMN, help="CT 地址列名称")
    parser.add_argument(
        "--details-csv",
        type=Path,
        help="可选：把逐行统计明细写入 CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excel_path = args.excel.expanduser().resolve()
    if not excel_path.is_file():
        print(f"错误：Excel 文件不存在：{excel_path}")
        return 1

    details: list[dict[str, object]] = []
    size_cache: dict[str, tuple[int, int, list[str]]] = {}
    unique_dirs: dict[str, Path] = {}
    reference_counts: dict[str, int] = {}
    row_total = 0
    nonempty_rows = 0
    matched_rows = 0
    missing_addresses = 0
    no_dicom_rows = 0
    error_count = 0

    try:
        for processed, (row_number, raw_address) in enumerate(
            iter_addresses(excel_path, args.sheet, args.column), start=1
        ):
            address = normalize_path(raw_address)
            row_size = 0
            row_files = 0
            dicom_dirs: list[Path] = []
            status = "OK"

            if address is None:
                status = "空地址"
            else:
                nonempty_rows += 1
                if not address.is_dir():
                    status = "地址不存在或不是目录"
                    missing_addresses += 1
                else:
                    dicom_dirs = find_dicom_dirs(address)
                    if not dicom_dirs:
                        status = "未找到 dicom 文件夹"
                        no_dicom_rows += 1
                    else:
                        matched_rows += 1
                        statuses: list[str] = []
                        for dicom_dir in dicom_dirs:
                            key = canonical_key(dicom_dir)
                            unique_dirs[key] = dicom_dir
                            reference_counts[key] = reference_counts.get(key, 0) + 1
                            if key not in size_cache:
                                size_cache[key] = directory_size(dicom_dir)
                                error_count += len(size_cache[key][2])
                            size, files, errors = size_cache[key]
                            row_size += size
                            row_files += files
                            if errors:
                                statuses.append(f"{len(errors)} 个项目读取失败")
                        if statuses:
                            status = "；".join(statuses)

            row_total += row_size
            details.append(
                {
                    "excel_row": row_number,
                    "address": "" if address is None else str(address),
                    "dicom_dirs": " | ".join(str(path) for path in dicom_dirs),
                    "bytes": row_size,
                    "files": row_files,
                    "status": status,
                }
            )
            if processed % 100 == 0:
                print(f"已处理 {processed} 行……")
    except (OSError, ValueError, StopIteration) as exc:
        print(f"错误：{exc}")
        return 1

    unique_total = sum(size_cache[key][0] for key in unique_dirs)
    unique_files = sum(size_cache[key][1] for key in unique_dirs)
    duplicate_references = sum(count - 1 for count in reference_counts.values())

    if args.details_csv:
        write_details(details, args.details_csv.expanduser().resolve())

    print("\n统计完成")
    print(f"Excel 数据行数：{len(details)}")
    print(f"非空地址行数：{nonempty_rows}")
    print(f"找到 dicom 的行数：{matched_rows}")
    print(f"不存在的地址数：{missing_addresses}")
    print(f"未找到 dicom 的行数：{no_dicom_rows}")
    print(f"唯一 dicom 文件夹数：{len(unique_dirs)}")
    print(f"重复引用次数：{duplicate_references}")
    print(f"文件读取错误数：{error_count}")
    print(f"逐行累计数据量：{row_total:,} 字节（{human_size(row_total)}）")
    print(
        f"去重后的实际总数据量：{unique_total:,} 字节"
        f"（{human_size(unique_total)}，{unique_total / 1_000_000_000:.3f} GB）"
    )
    print(f"去重后的文件数：{unique_files:,}")
    if args.details_csv:
        print(f"明细文件：{args.details_csv.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
