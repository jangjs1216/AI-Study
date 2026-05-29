from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


FACES = ("A", "BL", "BT", "BB", "BR", "C")
ROOT_SLOTS = ("A", "BL/BR", "BT/BB", "C")
ROOT_SLOT_LABELS = {
    "A": "A",
    "BL/BR": "BL / BR",
    "BT/BB": "BT / BB",
    "C": "C",
}
FACE_DRAW_ORDER = ("BT", "BL", "A", "BB", "BR", "C")
CATEGORIES = ("Defects", "Refined", "Filtered")
CATEGORY_ALIASES = {
    "defect": "Defects",
    "defects": "Defects",
    "refined": "Refined",
    "filtered": "Filtered",
}
ALL_VALUE = "All"
TIME_ALL_VALUE = "All"
MODE_TOTAL_PATCHES = "Total patches"
MODE_UNIQUE_IMEI = "Unique IMEI count"
PANEL_KEYS = ("left", "right")
PANEL_TITLES = {"left": "Left date", "right": "Right date"}
MAX_SCAN_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))


def default_config_path() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "SetVisualizer" / "config.json"
    return Path.home() / ".set_visualizer" / "config.json"

FACE_UNITS = {
    "BL": (2, 31),
    "BT": (20, 2),
    "A": (20, 31),
    "BB": (20, 2),
    "BR": (2, 31),
    "C": (20, 31),
}
GRID_SHAPES = {
    "A": (31, 20),
    "C": (31, 20),
    "BL": (31, 2),
    "BR": (31, 2),
    "BT": (2, 21),
    "BB": (2, 21),
}
FILE_COORD_LIMITS = {
    "A": (19, 30),
    "C": (19, 30),
    "BL": (1, 30),
    "BR": (1, 30),
    "BT": (20, 1),
    "BB": (20, 1),
}
PANEL_COLORS = {
    "A": "#f8fafc",
    "C": "#eefdf7",
    "BL": "#fff7ed",
    "BR": "#fff7ed",
    "BT": "#eff6ff",
    "BB": "#eff6ff",
}
CATEGORY_LOOKUP = CATEGORY_ALIASES
DATE_FOLDER_RE = re.compile(r"^\d{6}$")
PATCH_RE = re.compile(
    r"^\[(?P<row>\d+)\]\[(?P<col>\d+)\]\[(?P<defect_type>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$",
    re.IGNORECASE,
)
PREFIX_PATCH_RE = re.compile(
    r"^(?P<category>[^\[]+)\[(?P<row>\d+)\]\[(?P<col>\d+)\]\[(?P<defect_type>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$",
    re.IGNORECASE,
)
EXTENDED_PATCH_RE = re.compile(
    r"^\[(?P<file_day>\d{6})\]\[(?P<file_imei>[^\]]+)\]\[(?P<row>\d+)\]\[(?P<col>\d+)\]"
    r"\[(?P<face_part>[^\]]+)\]\[(?P<position_part>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DateFolder:
    root_slot: str
    folder_name: str
    day: date
    path: Path


@dataclass(frozen=True)
class InspectionMeta:
    hhmmss: str
    imei: str
    model_file: str
    color_code: str


@dataclass(frozen=True)
class PatchMeta:
    category: str
    row: int
    col: int
    defect_type: str
    vector: str


@dataclass(frozen=True)
class PatchFileRef:
    category: str
    row: int
    col: int
    defect_type: str
    vector: str
    path: Path


@dataclass(frozen=True)
class PatchImage:
    face: str
    day_folder: str
    hhmmss: str
    imei: str
    model_file: str
    color_code: str
    category: str
    row: int
    col: int
    defect_type: str
    vector: str
    path: Path
    original_patch: PatchFileRef | None = None


@dataclass(frozen=True)
class ScanResult:
    patches: list[PatchImage]
    inspections: int
    imeis: tuple[str, ...]
    inspection_records: tuple[InspectionMeta, ...]
    errors: Counter[str]
    missing_faces: tuple[str, ...]


def normalize_path_for_cache(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def make_scan_cache_key(root_paths: dict[str, Path], selected_yymmdd: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    roots = tuple((root_slot, normalize_path_for_cache(root_paths[root_slot])) for root_slot in ROOT_SLOTS)
    return selected_yymmdd, roots


def load_saved_roots(config_path: Path | None = None) -> dict[str, str]:
    path = config_path or default_config_path()
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    roots = payload.get("roots")
    if not isinstance(roots, dict):
        return {}
    return {
        root_slot: str(roots.get(root_slot, "")).strip()
        for root_slot in ROOT_SLOTS
        if str(roots.get(root_slot, "")).strip()
    }


def save_roots(root_paths: dict[str, Path], config_path: Path | None = None) -> bool:
    path = config_path or default_config_path()
    payload = {"roots": {root_slot: str(root_paths[root_slot]) for root_slot in ROOT_SLOTS}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError:
        return False
    return True


def parse_date_folder_name(name: str) -> date | None:
    if not DATE_FOLDER_RE.fullmatch(name):
        return None
    try:
        return datetime.strptime(name, "%y%m%d").date()
    except ValueError:
        return None


def format_date_option(folder_name: str, day: date) -> str:
    return f"{folder_name} ({day.isoformat()})"


def parse_inspection_folder_name(name: str) -> InspectionMeta | None:
    try:
        head, color_code = name.rsplit("_", 1)
        hhmmss, imei, model_file = head.split("_", 2)
    except ValueError:
        return None

    if not re.fullmatch(r"\d{6}", hhmmss):
        return None
    if not imei or not model_file or len(color_code) != 2:
        return None
    return InspectionMeta(hhmmss=hhmmss, imei=imei, model_file=model_file, color_code=color_code)


def parse_patch_filename(name: str, category: str | None = None) -> PatchMeta | None:
    match = PATCH_RE.fullmatch(name)
    parsed_category = category
    if match is None:
        extended_match = EXTENDED_PATCH_RE.fullmatch(name)
        if extended_match is not None:
            if parsed_category is None:
                return None
            canonical_category = CATEGORY_LOOKUP.get(parsed_category.lower())
            if canonical_category is None:
                return None
            defect_type = f"{extended_match.group('face_part')}_{extended_match.group('position_part')}"
            return PatchMeta(
                category=canonical_category,
                row=int(extended_match.group("row")),
                col=int(extended_match.group("col")),
                defect_type=defect_type,
                vector=extended_match.group("vector"),
            )

        match = PREFIX_PATCH_RE.fullmatch(name)
        if match is None:
            return None
        parsed_category = match.group("category")

    if parsed_category is None:
        return None

    canonical_category = CATEGORY_LOOKUP.get(parsed_category.lower())
    if canonical_category is None:
        return None

    return PatchMeta(
        category=canonical_category,
        row=int(match.group("row")),
        col=int(match.group("col")),
        defect_type=match.group("defect_type"),
        vector=match.group("vector"),
    )


def to_display_cell(face: str, meta: PatchMeta) -> tuple[int, int] | None:
    primary = xy_to_display_cell(face, meta.row, meta.col)
    if primary is not None:
        return primary
    return xy_to_display_cell(face, meta.col, meta.row)


def xy_to_display_cell(face: str, x: int, y: int) -> tuple[int, int] | None:
    max_file_x, max_file_y = FILE_COORD_LIMITS[face]
    if not (0 <= x <= max_file_x and 0 <= y <= max_file_y):
        return None

    display_row = y
    display_col = x

    row_count, col_count = GRID_SHAPES[face]
    if not (0 <= display_row < row_count and 0 <= display_col < col_count):
        return None
    return display_row, display_col


def normalize_type_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def face_from_patch_type(root_slot: str, defect_type: str) -> str | None:
    if root_slot in {"A", "C"}:
        return root_slot

    type_key = normalize_type_key(defect_type)
    if root_slot == "BL/BR":
        if type_key.startswith("bleft"):
            return "BL"
        if type_key.startswith("bright"):
            return "BR"
    elif root_slot == "BT/BB":
        if type_key.startswith("btop"):
            return "BT"
        if type_key.startswith("bbottom"):
            return "BB"
    return None


def canonical_patch_face_part(face: str) -> str:
    return {
        "A": "A",
        "C": "C",
        "BL": "BLeft",
        "BR": "BRight",
        "BT": "BTop",
        "BB": "BBottom",
    }[face]


def split_patch_type_for_patches(face: str, defect_type: str) -> tuple[str, str]:
    face_part = canonical_patch_face_part(face)
    parts = [part for part in defect_type.split("_") if part]
    face_key = normalize_type_key(face_part)

    if parts:
        first_key = normalize_type_key(parts[0])
        if first_key == face_key:
            return face_part, "_".join(parts[1:]) or "Center"

    if len(parts) >= 2:
        first_two_key = normalize_type_key(parts[0] + parts[1])
        if first_two_key == face_key:
            return face_part, "_".join(parts[2:]) or "Center"

    type_key = normalize_type_key(defect_type)
    if type_key.startswith(face_key):
        suffix = type_key[len(face_key) :]
        return face_part, suffix[:1].upper() + suffix[1:] if suffix else "Center"

    if len(parts) >= 2:
        return face_part, "_".join(parts[1:])
    return face_part, "Center"


def build_patches_filename(day_folder: str, imei: str, face: str, meta: PatchMeta) -> str:
    face_part, position_part = split_patch_type_for_patches(face, meta.defect_type)
    return f"[{day_folder}][{imei}][{meta.row}][{meta.col}][{face_part}][{position_part}][{meta.vector}].png"


def build_original_patch_ref(
    inspection_dir: Path,
    day_folder: str,
    inspection: InspectionMeta,
    face: str,
    meta: PatchMeta,
) -> PatchFileRef:
    patches_filename = build_patches_filename(day_folder, inspection.imei, face, meta)
    face_part, position_part = split_patch_type_for_patches(face, meta.defect_type)
    return PatchFileRef(
        category="Patches",
        row=meta.row,
        col=meta.col,
        defect_type=f"{face_part}_{position_part}",
        vector=meta.vector,
        path=inspection_dir / "Patches" / patches_filename,
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def split_patch_type_preserving_prefix(defect_type: str) -> tuple[str, str] | None:
    parts = [part for part in defect_type.split("_") if part]
    if len(parts) < 2:
        return None
    return parts[0], "_".join(parts[1:]) or "Center"


def original_patch_path_candidates(patch: PatchImage) -> list[Path]:
    if patch.original_patch is None:
        return []

    patches_dir = patch.original_patch.path.parent
    paths = [patch.original_patch.path]
    ref = patch.original_patch

    preserved = split_patch_type_preserving_prefix(patch.defect_type)
    if preserved is not None:
        face_part, position_part = preserved
        paths.append(
            patches_dir
            / f"[{patch.day_folder}][{patch.imei}][{ref.row}][{ref.col}][{face_part}][{position_part}][{ref.vector}].png"
        )

    paths.append(patches_dir / f"[{ref.row}][{ref.col}][{patch.defect_type}][{ref.vector}].png")
    paths.append(patches_dir / f"Patches[{ref.row}][{ref.col}][{patch.defect_type}][{ref.vector}].png")
    return unique_paths(paths)


def normalize_category_name(value: str) -> str | None:
    return CATEGORY_LOOKUP.get(value.strip().lower())


def safe_is_dir(path: Path, errors: Counter[str], error_key: str) -> bool:
    try:
        return path.is_dir()
    except OSError:
        errors[error_key] += 1
        return False


def safe_is_file(path: Path, errors: Counter[str], error_key: str) -> bool:
    try:
        return path.is_file()
    except OSError:
        errors[error_key] += 1
        return False


def safe_list_files(folder: Path, errors: Counter[str], error_key: str) -> list[Path]:
    try:
        return [path for path in folder.iterdir() if safe_is_file(path, errors, error_key)]
    except OSError:
        errors[error_key] += 1
        return []


def safe_rglob_files(folder: Path, errors: Counter[str], error_key: str) -> list[Path]:
    files: list[Path] = []
    try:
        for path in folder.rglob("*"):
            if safe_is_file(path, errors, error_key):
                files.append(path)
    except OSError:
        errors[error_key] += 1
    return files


def scan_date_folders(root_paths: dict[str, Path]) -> tuple[list[DateFolder], Counter[str]]:
    folders: list[DateFolder] = []
    errors: Counter[str] = Counter()

    for root_slot, root in root_paths.items():
        if not root.is_dir():
            errors[f"{root_slot}: missing root"] += 1
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            errors[f"{root_slot}: root read error"] += 1
            continue

        for child in children:
            if not child.is_dir():
                continue
            parsed_day = parse_date_folder_name(child.name)
            if parsed_day is None:
                continue
            folders.append(DateFolder(root_slot=root_slot, folder_name=child.name, day=parsed_day, path=child))

    folders.sort(key=lambda item: (item.day, item.root_slot))
    return folders, errors


def scan_patches(root_paths: dict[str, Path], selected_yymmdd: str) -> ScanResult:
    patches: list[PatchImage] = []
    errors: Counter[str] = Counter()
    missing_roots: list[str] = []
    imeis: set[str] = set()
    inspection_records: set[InspectionMeta] = set()
    inspections = 0
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as executor:
        for root_slot, root in root_paths.items():
            date_path = root / selected_yymmdd
            if not date_path.is_dir():
                missing_roots.append(root_slot)
                continue

            try:
                inspection_dirs = [path for path in date_path.iterdir() if path.is_dir()]
            except OSError:
                errors[f"{root_slot}: date read error"] += 1
                continue

            for inspection_dir in inspection_dirs:
                inspection = parse_inspection_folder_name(inspection_dir.name)
                if inspection is None:
                    errors["invalid inspection folder"] += 1
                    continue
                imeis.add(inspection.imei)
                inspection_records.add(inspection)
                inspections += 1
                futures.append(
                    executor.submit(
                        _scan_inspection_patches,
                        root_slot,
                        selected_yymmdd,
                        inspection_dir,
                        inspection,
                    )
                )

        for future in as_completed(futures):
            try:
                worker_patches, worker_errors = future.result()
            except OSError as exc:
                errors[f"scan worker os error: {type(exc).__name__}"] += 1
                continue
            except Exception as exc:
                errors[f"scan worker error: {type(exc).__name__}"] += 1
                continue
            patches.extend(worker_patches)
            errors.update(worker_errors)

    return ScanResult(
        patches=patches,
        inspections=inspections,
        imeis=tuple(sorted(imeis)),
        inspection_records=tuple(
            sorted(
                inspection_records,
                key=lambda item: (item.hhmmss, item.imei, item.model_file, item.color_code),
            )
        ),
        errors=errors,
        missing_faces=tuple(missing_roots),
    )


def _scan_inspection_patches(
    root_slot: str,
    day_folder: str,
    inspection_dir: Path,
    inspection: InspectionMeta,
) -> tuple[list[PatchImage], Counter[str]]:
    patches: list[PatchImage] = []
    errors: Counter[str] = Counter()

    try:
        inspection_children = list(inspection_dir.iterdir())
    except OSError:
        errors["inspection read error"] += 1
        return patches, errors

    category_dirs: dict[str, Path] = {}
    direct_files: list[Path] = []
    for child in inspection_children:
        if safe_is_dir(child, errors, "inspection child stat error"):
            category = normalize_category_name(child.name)
            if category is not None:
                category_dirs[category] = child
        elif safe_is_file(child, errors, "inspection child stat error"):
            direct_files.append(child)

    for category in CATEGORIES:
        category_dir = category_dirs.get(category)
        if category_dir is None:
            continue

        category_files = safe_list_files(category_dir, errors, f"{category}: read error")

        if not category_files:
            category_files = safe_rglob_files(category_dir, errors, f"{category}: recursive read error")

        for path in category_files:
            patch = _build_patch_image(root_slot, day_folder, inspection_dir, inspection, path, errors, category)
            if patch is not None:
                patches.append(patch)

    for path in direct_files:
        patch = _build_patch_image(root_slot, day_folder, inspection_dir, inspection, path, errors, None)
        if patch is not None:
            patches.append(patch)

    return patches, errors


def _build_patch_image(
    root_slot: str,
    day_folder: str,
    inspection_dir: Path,
    inspection: InspectionMeta,
    path: Path,
    errors: Counter[str],
    category: str | None,
) -> PatchImage | None:
    if path.suffix.lower() != ".png":
        errors["non png file"] += 1
        return None

    meta = parse_patch_filename(path.name, category)
    if meta is None:
        errors["invalid patch filename"] += 1
        return None

    face = face_from_patch_type(root_slot, meta.defect_type)
    if face is None:
        errors["patch type root mismatch"] += 1
        return None

    display_cell = to_display_cell(face, meta)
    if display_cell is None:
        errors["patch out of grid"] += 1
        return None
    display_row, display_col = display_cell
    original_patch = build_original_patch_ref(inspection_dir, day_folder, inspection, face, meta)

    return PatchImage(
        face=face,
        day_folder=day_folder,
        hhmmss=inspection.hhmmss,
        imei=inspection.imei,
        model_file=inspection.model_file,
        color_code=inspection.color_code,
        category=meta.category,
        row=display_row,
        col=display_col,
        defect_type=meta.defect_type,
        vector=meta.vector,
        path=path,
        original_patch=original_patch,
    )


def make_empty_counts() -> dict[str, list[list[int]]]:
    return {face: [[0 for _ in range(cols)] for _ in range(rows)] for face, (rows, cols) in GRID_SHAPES.items()}


def filter_patches(patches: list[PatchImage], category: str, imei: str) -> list[PatchImage]:
    result = patches
    if category != ALL_VALUE:
        result = [patch for patch in result if patch.category == category]
    if imei != ALL_VALUE:
        result = [patch for patch in result if patch.imei == imei]
    return result


def format_time_option(hhmmss: str) -> str:
    if not re.fullmatch(r"\d{6}", hhmmss):
        return hhmmss
    return f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}"


def parse_time_option(value: str) -> str | None:
    text = value.strip()
    if not text or text == TIME_ALL_VALUE:
        return None
    digits = re.sub(r"\D", "", text)
    if re.fullmatch(r"\d{6}", digits):
        return digits
    return None


def normalize_time_range(start: str | None, end: str | None) -> tuple[str | None, str | None]:
    if start is not None and end is not None and start > end:
        return end, start
    return start, end


def is_time_in_range(hhmmss: str, start: str | None, end: str | None) -> bool:
    if start is not None and hhmmss < start:
        return False
    if end is not None and hhmmss > end:
        return False
    return True


def filter_patches_by_time(
    patches: list[PatchImage],
    start: str | None,
    end: str | None,
) -> list[PatchImage]:
    start, end = normalize_time_range(start, end)
    if start is None and end is None:
        return patches
    return [patch for patch in patches if is_time_in_range(patch.hhmmss, start, end)]


def filter_inspection_records_by_time(
    records: tuple[InspectionMeta, ...],
    start: str | None,
    end: str | None,
) -> tuple[InspectionMeta, ...]:
    start, end = normalize_time_range(start, end)
    if start is None and end is None:
        return records
    return tuple(record for record in records if is_time_in_range(record.hhmmss, start, end))


def aggregate_total_patches(patches: list[PatchImage]) -> dict[str, list[list[int]]]:
    counts = make_empty_counts()
    for patch in patches:
        counts[patch.face][patch.row][patch.col] += 1
    return counts


def aggregate_unique_imeis(patches: list[PatchImage]) -> dict[str, list[list[int]]]:
    seen: dict[str, list[list[set[str]]]] = {
        face: [[set() for _ in range(cols)] for _ in range(rows)] for face, (rows, cols) in GRID_SHAPES.items()
    }
    for patch in patches:
        seen[patch.face][patch.row][patch.col].add(patch.imei)
    return {
        face: [[len(imeis) for imeis in row] for row in face_seen]
        for face, face_seen in seen.items()
    }


def summarize_imeis(patches: list[PatchImage]) -> list[tuple[str, int, int, int, int]]:
    by_imei: dict[str, Counter[str]] = defaultdict(Counter)
    for patch in patches:
        by_imei[patch.imei][patch.category] += 1

    rows: list[tuple[str, int, int, int, int]] = []
    for imei, counts in by_imei.items():
        defects = counts["Defects"]
        refined = counts["Refined"]
        filtered = counts["Filtered"]
        rows.append((imei, defects + refined + filtered, defects, refined, filtered))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def summarize_face_presence(
    patches: list[PatchImage],
    denominator_imeis: set[str],
) -> list[tuple[str, int, int, float, int, float, int, float]]:
    total = len(denominator_imeis)
    by_face_category: dict[str, dict[str, set[str]]] = {
        face: {category: set() for category in CATEGORIES} for face in FACES
    }

    for patch in patches:
        if patch.imei not in denominator_imeis:
            continue
        by_face_category[patch.face][patch.category].add(patch.imei)

    rows: list[tuple[str, int, int, float, int, float, int, float]] = []
    for face in FACES:
        defects = len(by_face_category[face]["Defects"])
        refined = len(by_face_category[face]["Refined"])
        filtered = len(by_face_category[face]["Filtered"])
        rows.append(
            (
                face,
                total,
                defects,
                defects / total if total else 0.0,
                refined,
                refined / total if total else 0.0,
                filtered,
                filtered / total if total else 0.0,
            )
        )
    return rows


def format_presence_rate(count: int, total: int, rate: float) -> str:
    return f"{count}/{total} ({rate * 100:.1f}%)"


def patch_sort_key(patch: PatchImage) -> tuple[str, str, str, str, int, int, str, str, str]:
    return (
        patch.imei,
        patch.category,
        patch.face,
        patch.hhmmss,
        patch.row,
        patch.col,
        patch.defect_type,
        patch.vector,
        str(patch.path),
    )


def max_count(counts: dict[str, list[list[int]]]) -> int:
    return max((value for face_counts in counts.values() for row in face_counts for value in row), default=0)


def interpolate_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    start_values = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_values = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(round(start_values[i] + (end_values[i] - start_values[i]) * ratio) for i in range(3))
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def heat_color(value: int, max_value: int, fallback: str) -> str:
    if value <= 0 or max_value <= 0:
        return fallback
    ratio = value / max_value
    if ratio < 0.5:
        return interpolate_color("#dbeafe", "#60a5fa", ratio / 0.5)
    return interpolate_color("#60a5fa", "#dc2626", (ratio - 0.5) / 0.5)


def compute_face_layout(
    canvas_width: int,
    canvas_height: int,
    padding: int = 28,
    gap: int = 8,
) -> dict[str, Rect]:
    total_unit_width = 2 + 20 + 2 + 20
    total_unit_height = 2 + 31 + 2
    available_width = max(1, canvas_width - padding * 2 - gap * 3)
    available_height = max(1, canvas_height - padding * 2 - gap * 2)
    unit = min(available_width / total_unit_width, available_height / total_unit_height)

    total_width = total_unit_width * unit + gap * 3
    total_height = total_unit_height * unit + gap * 2
    left = (canvas_width - total_width) / 2
    top = (canvas_height - total_height) / 2

    bl_width = FACE_UNITS["BL"][0] * unit
    a_width = FACE_UNITS["A"][0] * unit
    br_width = FACE_UNITS["BR"][0] * unit
    c_width = FACE_UNITS["C"][0] * unit
    bt_height = FACE_UNITS["BT"][1] * unit
    a_height = FACE_UNITS["A"][1] * unit
    bb_height = FACE_UNITS["BB"][1] * unit

    x_bl = left
    x_a = x_bl + bl_width + gap
    x_br = x_a + a_width + gap
    x_c = x_br + br_width + gap
    y_bt = top
    y_main = y_bt + bt_height + gap
    y_bb = y_main + a_height + gap

    return {
        "BT": Rect(x_a, y_bt, x_a + a_width, y_bt + bt_height),
        "BL": Rect(x_bl, y_main, x_bl + bl_width, y_main + a_height),
        "A": Rect(x_a, y_main, x_a + a_width, y_main + a_height),
        "BB": Rect(x_a, y_bb, x_a + a_width, y_bb + bb_height),
        "BR": Rect(x_br, y_main, x_br + br_width, y_main + a_height),
        "C": Rect(x_c, y_main, x_c + c_width, y_main + a_height),
    }


class PhoneSetCanvas(ttk.Frame):
    def __init__(self, parent: tk.Widget, cell_click_callback=None) -> None:
        super().__init__(parent)
        self.counts = make_empty_counts()
        self.max_value = 0
        self.cell_click_callback = cell_click_callback
        self.canvas = tk.Canvas(self, bg="#f1f5f9", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self.on_click)

    def set_counts(self, counts: dict[str, list[list[int]]]) -> None:
        self.counts = counts
        self.max_value = max_count(counts)
        self.draw()

    def draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        layout = compute_face_layout(width, height)

        self._draw_alignment_guides(layout)
        for face in FACE_DRAW_ORDER:
            self._draw_face(face, layout[face])

    def _draw_alignment_guides(self, layout: dict[str, Rect]) -> None:
        a = layout["A"]
        c = layout["C"]
        self.canvas.create_line(a.x1, a.y1, c.x2, a.y1, fill="#cbd5e1", dash=(3, 5))
        self.canvas.create_line(a.x1, a.y2, c.x2, a.y2, fill="#cbd5e1", dash=(3, 5))

    def _draw_face(self, face: str, rect: Rect) -> None:
        rows, cols = GRID_SHAPES[face]
        cell_width = rect.width / cols
        cell_height = rect.height / rows
        fallback = PANEL_COLORS[face]
        font_size = max(6, min(9, int(min(cell_width, cell_height) * 0.42)))

        for row in range(rows):
            for col in range(cols):
                value = self.counts[face][row][col]
                x1 = rect.x1 + col * cell_width
                y1 = rect.y1 + row * cell_height
                x2 = rect.x1 + (col + 1) * cell_width
                y2 = rect.y1 + (row + 1) * cell_height
                ratio = value / self.max_value if self.max_value else 0
                fill = heat_color(value, self.max_value, fallback)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#cbd5e1")
                if value:
                    text_color = "#ffffff" if ratio >= 0.58 else "#0f172a"
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(value),
                        fill=text_color,
                        font=("", font_size, "bold"),
                    )

        outline = "#334155" if face in {"A", "C"} else "#64748b"
        self.canvas.create_rectangle(rect.x1, rect.y1, rect.x2, rect.y2, outline=outline, width=2)
        self.canvas.create_rectangle(rect.x1 + 2, rect.y1 + 2, rect.x1 + 36, rect.y1 + 20, fill="#ffffff", outline="")
        self.canvas.create_text(rect.x1 + 7, rect.y1 + 5, text=face, anchor="nw", fill="#0f172a", font=("", 9, "bold"))

    def on_click(self, event: tk.Event) -> None:
        if self.cell_click_callback is None:
            return
        cell = self.hit_test(event.x, event.y)
        if cell is None:
            return
        face, row, col = cell
        if self.counts[face][row][col] <= 0:
            return
        self.cell_click_callback(face, row, col)

    def hit_test(self, x: int, y: int) -> tuple[str, int, int] | None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        layout = compute_face_layout(width, height)
        for face in FACE_DRAW_ORDER:
            rect = layout[face]
            if not (rect.x1 <= x <= rect.x2 and rect.y1 <= y <= rect.y2):
                continue
            rows, cols = GRID_SHAPES[face]
            col = min(cols - 1, max(0, int((x - rect.x1) / (rect.width / cols))))
            row = min(rows - 1, max(0, int((y - rect.y1) / (rect.height / rows))))
            return face, row, col
        return None


class PatchViewerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("Patch Viewer")
        self.geometry("1180x720")
        self.minsize(900, 560)
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        self.patches: list[PatchImage] = []
        self.patch_by_iid: dict[str, PatchImage] = {}
        self.image_refs: list[tk.PhotoImage] = []
        self.info_var = tk.StringVar(value="No patch selected")

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        columns = ("imei", "face", "category", "row", "col", "type", "vector", "time")
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=8)
        headings = {
            "imei": "IMEI",
            "face": "Face",
            "category": "Category",
            "row": "Row",
            "col": "Col",
            "type": "Type",
            "vector": "Vector",
            "time": "Time",
        }
        widths = {"imei": 170, "face": 52, "category": 76, "row": 52, "col": 52, "type": 150, "vector": 64, "time": 72}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.on_patch_select)

        ttk.Label(root, textvariable=self.info_var, anchor="w").pack(fill="x", pady=(6, 8))

        image_area = ttk.Frame(root)
        image_area.pack(fill="both", expand=True)
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=1)
        image_area.rowconfigure(1, weight=1)

        ttk.Label(image_area, text="Patch Image", font=("", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(image_area, text="Original Patch", font=("", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.patch_image_label = ttk.Label(image_area, anchor="center", relief="sunken")
        self.original_image_label = ttk.Label(image_area, anchor="center", relief="sunken")
        self.patch_image_label.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.original_image_label.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(4, 0))

        self.patch_path_var = tk.StringVar(value="")
        self.original_path_var = tk.StringVar(value="")
        ttk.Label(image_area, textvariable=self.patch_path_var, anchor="w").grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(image_area, textvariable=self.original_path_var, anchor="w").grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(10, 0),
            pady=(4, 0),
        )

    def show_patches(self, panel_title: str, patches: list[PatchImage]) -> None:
        self.deiconify()
        self.lift()
        self.patches = sorted(patches, key=patch_sort_key)
        self.patch_by_iid.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.title(f"Patch Viewer - {panel_title}")
        for index, patch in enumerate(self.patches):
            iid = str(index)
            self.patch_by_iid[iid] = patch
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    patch.imei,
                    patch.face,
                    patch.category,
                    patch.row,
                    patch.col,
                    patch.defect_type,
                    patch.vector,
                    patch.hhmmss,
                ),
            )

        if self.patches:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.display_patch(self.patches[0])
        else:
            self.clear_images("No patches in selected cell")

    def on_patch_select(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        patch = self.patch_by_iid.get(selected[0])
        if patch is not None:
            self.display_patch(patch)

    def display_patch(self, patch: PatchImage) -> None:
        self.info_var.set(
            f"{patch.day_folder} {patch.hhmmss} | IMEI={patch.imei} | model={patch.model_file} | "
            f"color={patch.color_code} | {patch.face}[{patch.row}][{patch.col}] | {patch.category}"
        )
        self.patch_path_var.set(str(patch.path))
        original_path = self.resolve_original_patch_path(patch)
        if original_path is not None:
            self.original_path_var.set(str(original_path))
        elif patch.original_patch is not None:
            self.original_path_var.set(f"Expected: {patch.original_patch.path}")
        else:
            self.original_path_var.set("No original patch metadata")

        self.image_refs.clear()
        self._set_image(self.patch_image_label, patch.path, "Patch image not available")
        self._set_image(self.original_image_label, original_path, "Original patch not available")

    def clear_images(self, message: str) -> None:
        self.info_var.set(message)
        self.patch_path_var.set("")
        self.original_path_var.set("")
        self.image_refs.clear()
        for label in (self.patch_image_label, self.original_image_label):
            label.configure(image="", text=message)

    def _set_image(self, label: ttk.Label, path: Path | None, missing_text: str) -> None:
        image, message = self._load_photo_image(path)
        if image is None:
            label.configure(image="", text=message or missing_text)
            return
        self.image_refs.append(image)
        label.configure(image=image, text="")

    def resolve_original_patch_path(self, patch: PatchImage) -> Path | None:
        for path in original_patch_path_candidates(patch):
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
        return None

    def _load_photo_image(
        self,
        path: Path | None,
        max_size: tuple[int, int] = (520, 460),
    ) -> tuple[tk.PhotoImage | None, str]:
        if path is None:
            return None, "No original patch metadata"
        try:
            if not path.is_file():
                return None, "Image file not found"
        except OSError as exc:
            return None, f"Image stat failed: {type(exc).__name__}"

        try:
            if Image is not None and ImageTk is not None:
                with Image.open(path) as image:
                    resampling = getattr(Image, "Resampling", Image)
                    image.thumbnail(max_size, getattr(resampling, "LANCZOS", 1))
                    return ImageTk.PhotoImage(image.copy()), ""

            image = tk.PhotoImage(file=str(path))
            factor = max(
                1,
                (image.width() + max_size[0] - 1) // max_size[0],
                (image.height() + max_size[1] - 1) // max_size[1],
            )
            if factor > 1:
                image = image.subsample(factor, factor)
            return image, ""
        except Exception as exc:
            return None, f"Image load failed: {type(exc).__name__}"


class SetVisualizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Set Visualizer")
        self.geometry("1720x900")
        self.minsize(1280, 760)

        self.root_vars = {root_slot: tk.StringVar() for root_slot in ROOT_SLOTS}
        self.date_vars = {key: tk.StringVar() for key in PANEL_KEYS}
        self.time_start_vars = {key: tk.StringVar(value=TIME_ALL_VALUE) for key in PANEL_KEYS}
        self.time_end_vars = {key: tk.StringVar(value=TIME_ALL_VALUE) for key in PANEL_KEYS}
        self.category_var = tk.StringVar(value=ALL_VALUE)
        self.mode_var = tk.StringVar(value=MODE_TOTAL_PATCHES)
        self.imei_var = tk.StringVar(value=ALL_VALUE)
        self.status_var = tk.StringVar(value="Ready")
        self.detail_vars = {key: tk.StringVar(value="No date loaded") for key in PANEL_KEYS}
        self.date_options: dict[str, str] = {}
        self.panel_patches: dict[str, list[PatchImage]] = {key: [] for key in PANEL_KEYS}
        self.panel_imeis: dict[str, set[str]] = {key: set() for key in PANEL_KEYS}
        self.panel_inspection_records: dict[str, tuple[InspectionMeta, ...]] = {key: () for key in PANEL_KEYS}
        self.panel_errors: dict[str, Counter[str]] = {key: Counter() for key in PANEL_KEYS}
        self.panel_missing_faces: dict[str, tuple[str, ...]] = {key: () for key in PANEL_KEYS}
        self.panel_inspections: dict[str, int] = {key: 0 for key in PANEL_KEYS}
        self.date_combos: dict[str, ttk.Combobox] = {}
        self.time_start_combos: dict[str, ttk.Combobox] = {}
        self.time_end_combos: dict[str, ttk.Combobox] = {}
        self.apply_buttons: dict[str, ttk.Button] = {}
        self.canvases: dict[str, PhoneSetCanvas] = {}
        self.summary_trees: dict[str, ttk.Treeview] = {}
        self.face_rate_trees: dict[str, ttk.Treeview] = {}
        self.scan_cache: dict[tuple[str, tuple[tuple[str, str], ...]], ScanResult] = {}
        self.pending_scan_panels: dict[tuple[str, tuple[tuple[str, str], ...]], set[str]] = {}
        self.panel_scan_keys: dict[str, tuple[str, tuple[tuple[str, str], ...]] | None] = {
            key: None for key in PANEL_KEYS
        }
        self.patch_viewers: dict[str, PatchViewerWindow | None] = {key: None for key in PANEL_KEYS}

        self._build_ui()
        self._set_date_controls_enabled(False)
        self._set_filter_controls_enabled(False)
        self.restore_saved_roots()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        path_frame = ttk.LabelFrame(root, text="Image roots", padding=8)
        path_frame.pack(fill="x")
        for index, root_slot in enumerate(ROOT_SLOTS):
            row = index // 2
            col = (index % 2) * 3
            ttk.Label(path_frame, text=ROOT_SLOT_LABELS[root_slot], width=7).grid(
                row=row,
                column=col,
                sticky="w",
                padx=(0, 4),
                pady=2,
            )
            ttk.Entry(path_frame, textvariable=self.root_vars[root_slot]).grid(
                row=row,
                column=col + 1,
                sticky="ew",
                padx=(0, 4),
                pady=2,
            )
            ttk.Button(path_frame, text="Browse", command=lambda item=root_slot: self.select_root(item)).grid(
                row=row,
                column=col + 2,
                sticky="ew",
                padx=(0, 10),
                pady=2,
            )
        path_frame.columnconfigure(1, weight=1)
        path_frame.columnconfigure(4, weight=1)

        action_frame = ttk.Frame(root)
        action_frame.pack(fill="x", pady=(8, 0))
        ttk.Button(action_frame, text="Done", command=self.load_dates).pack(side="left")

        ttk.Label(action_frame, text="Category").pack(side="left", padx=(14, 4))
        self.category_combo = ttk.Combobox(
            action_frame,
            textvariable=self.category_var,
            width=12,
            state="disabled",
            values=(ALL_VALUE, *CATEGORIES),
        )
        self.category_combo.pack(side="left", padx=(0, 12))

        ttk.Label(action_frame, text="Mode").pack(side="left", padx=(0, 4))
        self.mode_combo = ttk.Combobox(
            action_frame,
            textvariable=self.mode_var,
            width=18,
            state="disabled",
            values=(MODE_TOTAL_PATCHES, MODE_UNIQUE_IMEI),
        )
        self.mode_combo.pack(side="left", padx=(0, 12))

        ttk.Label(action_frame, text="IMEI").pack(side="left", padx=(0, 4))
        self.imei_combo = ttk.Combobox(action_frame, textvariable=self.imei_var, width=28, state="disabled", values=(ALL_VALUE,))
        self.imei_combo.pack(side="left")

        self.category_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visuals())
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visuals())
        self.imei_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visuals())

        body = ttk.PanedWindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 8))
        for key in PANEL_KEYS:
            panel = self._build_compare_panel(body, key)
            body.add(panel, weight=1)

        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x")

    def _build_compare_panel(self, parent: tk.Widget, key: str) -> ttk.Frame:
        panel = ttk.Frame(parent, padding=(0, 0, 8, 0) if key == "left" else (8, 0, 0, 0))

        header = ttk.Frame(panel)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text=PANEL_TITLES[key], font=("", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Date").pack(side="left", padx=(14, 4))
        date_combo = ttk.Combobox(header, textvariable=self.date_vars[key], width=22, state="disabled")
        date_combo.pack(side="left")
        apply_button = ttk.Button(
            header,
            text="Apply",
            command=lambda panel_key=key: self.apply_panel_date(panel_key),
            state="disabled",
        )
        apply_button.pack(side="left", padx=(6, 0))
        self.date_combos[key] = date_combo
        self.apply_buttons[key] = apply_button

        ttk.Label(header, text="Time").pack(side="left", padx=(12, 4))
        start_combo = ttk.Combobox(
            header,
            textvariable=self.time_start_vars[key],
            width=9,
            state="disabled",
            values=(TIME_ALL_VALUE,),
        )
        end_combo = ttk.Combobox(
            header,
            textvariable=self.time_end_vars[key],
            width=9,
            state="disabled",
            values=(TIME_ALL_VALUE,),
        )
        start_combo.pack(side="left")
        ttk.Label(header, text="~").pack(side="left", padx=3)
        end_combo.pack(side="left")
        start_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visuals())
        end_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visuals())
        self.time_start_combos[key] = start_combo
        self.time_end_combos[key] = end_combo

        canvas = PhoneSetCanvas(
            panel,
            cell_click_callback=lambda face, row, col, panel_key=key: self.on_cell_click(panel_key, face, row, col),
        )
        canvas.pack(fill="both", expand=True)
        self.canvases[key] = canvas

        summary_frame = ttk.Frame(panel)
        summary_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(summary_frame, text="IMEI Summary", font=("", 9, "bold")).pack(anchor="w")
        columns = ("imei", "total", "defects", "refined", "filtered")
        tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=6)
        headings = {
            "imei": "IMEI",
            "total": "Total",
            "defects": "Defects",
            "refined": "Refined",
            "filtered": "Filtered",
        }
        widths = {"imei": 170, "total": 58, "defects": 66, "refined": 66, "filtered": 66}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center")
        tree.pack(fill="x", pady=(3, 4))
        tree.bind("<<TreeviewSelect>>", lambda _event, panel_key=key: self.on_summary_select(panel_key))
        self.summary_trees[key] = tree

        rate_frame = ttk.Frame(panel)
        rate_frame.pack(fill="x", pady=(2, 0))
        ttk.Label(rate_frame, text="Face Presence Rate", font=("", 9, "bold")).pack(anchor="w")
        rate_columns = ("face", "defects", "refined", "filtered")
        rate_tree = ttk.Treeview(rate_frame, columns=rate_columns, show="headings", height=len(FACES))
        rate_headings = {
            "face": "Face",
            "defects": "Defects",
            "refined": "Refined",
            "filtered": "Filtered",
        }
        rate_widths = {"face": 58, "defects": 120, "refined": 120, "filtered": 120}
        for column in rate_columns:
            rate_tree.heading(column, text=rate_headings[column])
            rate_tree.column(column, width=rate_widths[column], anchor="center")
        rate_tree.pack(fill="x", pady=(3, 4))
        self.face_rate_trees[key] = rate_tree

        ttk.Label(panel, textvariable=self.detail_vars[key], justify="left", anchor="nw").pack(fill="x")
        return panel

    def select_root(self, root_slot: str) -> None:
        initial_dir = self.root_vars[root_slot].get().strip() or str(Path.cwd())
        selected = filedialog.askdirectory(
            title=f"Select {ROOT_SLOT_LABELS[root_slot]} root",
            initialdir=initial_dir,
        )
        if selected:
            self.root_vars[root_slot].set(selected)

    def get_root_paths(self) -> dict[str, Path] | None:
        roots: dict[str, Path] = {}
        missing = []
        for root_slot in ROOT_SLOTS:
            text = self.root_vars[root_slot].get().strip()
            if not text:
                missing.append(ROOT_SLOT_LABELS[root_slot])
            roots[root_slot] = Path(text)
        if missing:
            messagebox.showwarning("Missing roots", f"Set root paths for: {', '.join(missing)}")
            return None
        return roots

    def restore_saved_roots(self) -> None:
        saved_roots = load_saved_roots()
        if not saved_roots:
            return
        for root_slot, path_text in saved_roots.items():
            self.root_vars[root_slot].set(path_text)
        if all(self.root_vars[root_slot].get().strip() for root_slot in ROOT_SLOTS):
            self.status_var.set("Loaded saved image roots")
            self.after(150, self.load_dates)

    def load_dates(self) -> None:
        roots = self.get_root_paths()
        if roots is None:
            return

        saved = save_roots(roots)
        self.scan_cache.clear()
        self.pending_scan_panels.clear()
        for key in PANEL_KEYS:
            self.panel_scan_keys[key] = None

        folders, errors = scan_date_folders(roots)
        by_folder: dict[str, date] = {}
        for folder in folders:
            by_folder[folder.folder_name] = folder.day

        self.date_options = {
            format_date_option(folder_name, day): folder_name
            for folder_name, day in sorted(by_folder.items(), key=lambda item: item[1])
        }
        values = tuple(self.date_options.keys())
        for combo in self.date_combos.values():
            combo.configure(values=values)
        if values:
            left_value = values[-2] if len(values) >= 2 else values[-1]
            self.date_vars["left"].set(left_value)
            self.date_vars["right"].set(values[-1])
            self._set_date_controls_enabled(True)
            self.status_var.set(f"Loaded {len(values)} date folders")
        else:
            for key in PANEL_KEYS:
                self.date_vars[key].set("")
            self._set_date_controls_enabled(False)
            self.status_var.set("No yymmdd date folders found")

        if errors:
            self.status_var.set(f"{self.status_var.get()} | {self.format_errors(errors)}")
        if not saved:
            self.status_var.set(f"{self.status_var.get()} | root save failed")

    def apply_panel_date(self, key: str) -> None:
        roots = self.get_root_paths()
        if roots is None:
            return
        selected = self.date_options.get(self.date_vars[key].get())
        if not selected:
            messagebox.showwarning("Missing date", "Select a date folder first")
            return

        cache_key = make_scan_cache_key(roots, selected)
        self.panel_scan_keys[key] = cache_key
        cached = self.scan_cache.get(cache_key)
        if cached is not None:
            self.apply_scan_result((key,), cached)
            self.status_var.set(f"Loaded {selected} from cache")
            return

        if cache_key in self.pending_scan_panels:
            self.pending_scan_panels[cache_key].add(key)
            self.apply_buttons[key].configure(state="disabled")
            self.status_var.set(f"Waiting for existing scan: {selected}")
            return

        self.pending_scan_panels[cache_key] = {key}
        self.apply_buttons[key].configure(state="disabled")
        self.status_var.set(f"Scanning {selected} with up to {MAX_SCAN_WORKERS} workers...")
        thread = threading.Thread(
            target=self.scan_panel_worker,
            args=(cache_key, roots, selected),
            daemon=True,
        )
        thread.start()

    def scan_panel_worker(
        self,
        cache_key: tuple[str, tuple[tuple[str, str], ...]],
        roots: dict[str, Path],
        selected: str,
    ) -> None:
        try:
            result = scan_patches(roots, selected)
            error: Exception | None = None
        except Exception as exc:
            result = ScanResult(
                patches=[],
                inspections=0,
                imeis=(),
                inspection_records=(),
                errors=Counter({"scan failed": 1}),
                missing_faces=(),
            )
            error = exc
        try:
            self.after(0, lambda: self.finish_panel_scan(cache_key, result, error))
        except RuntimeError:
            return

    def finish_panel_scan(
        self,
        cache_key: tuple[str, tuple[tuple[str, str], ...]],
        result: ScanResult,
        error: Exception | None,
    ) -> None:
        panels = self.pending_scan_panels.pop(cache_key, None)
        if panels is None:
            return
        if error is None:
            self.scan_cache[cache_key] = result
        active_panels = tuple(panel for panel in panels if self.panel_scan_keys.get(panel) == cache_key)
        self.apply_scan_result(active_panels, result)
        for panel in panels:
            self.apply_buttons[panel].configure(state="normal")

        selected = cache_key[0]
        if error is not None:
            self.status_var.set(f"Scan failed for {selected}: {error}")
        else:
            self.status_var.set(
                f"Scan complete for {selected}: inspections={result.inspections:,}, patches={len(result.patches):,}"
            )

    def apply_scan_result(self, panels: tuple[str, ...], result: ScanResult) -> None:
        for panel in panels:
            self.panel_patches[panel] = result.patches
            self.panel_imeis[panel] = set(result.imeis)
            self.panel_inspection_records[panel] = result.inspection_records
            self.panel_errors[panel] = result.errors
            self.panel_missing_faces[panel] = result.missing_faces
            self.panel_inspections[panel] = result.inspections
            self.populate_time_choices(panel)
        if panels:
            self.populate_imei_choices()
            self._set_filter_controls_enabled(True)
            self.refresh_visuals()

    def populate_imei_choices(self) -> None:
        current = self.imei_var.get()
        imeis = sorted(
            {
                imei
                for panel_imeis in self.panel_imeis.values()
                for imei in panel_imeis
            }
        )
        values = (ALL_VALUE, *imeis)
        self.imei_combo.configure(values=values)
        if current in values:
            self.imei_var.set(current)
        else:
            self.imei_var.set(ALL_VALUE)

    def populate_time_choices(self, key: str) -> None:
        records = self.panel_inspection_records[key]
        values = (TIME_ALL_VALUE, *tuple(format_time_option(record.hhmmss) for record in records))
        values = tuple(dict.fromkeys(values))

        self.time_start_combos[key].configure(values=values)
        self.time_end_combos[key].configure(values=values)
        if self.time_start_vars[key].get() not in values:
            self.time_start_vars[key].set(TIME_ALL_VALUE)
        if self.time_end_vars[key].get() not in values:
            self.time_end_vars[key].set(TIME_ALL_VALUE)

        state = "readonly" if len(values) > 1 else "disabled"
        self.time_start_combos[key].configure(state=state)
        self.time_end_combos[key].configure(state=state)

    def refresh_visuals(self) -> None:
        category = self.category_var.get() or ALL_VALUE
        imei = self.imei_var.get() or ALL_VALUE

        panel_parts: list[str] = []
        for key in PANEL_KEYS:
            patches = self.current_filtered_patches(key)
            if self.mode_var.get() == MODE_UNIQUE_IMEI:
                counts = aggregate_unique_imeis(patches)
            else:
                counts = aggregate_total_patches(patches)
            self.canvases[key].set_counts(counts)
            self.populate_summary(key)
            self.populate_face_rate(key)
            panel_parts.append(self.update_panel_status(key, patches, counts))

        self.status_var.set(
            f"category={category} | mode={self.mode_var.get()} | imei={imei} | " + " | ".join(panel_parts)
        )

    def populate_summary(self, key: str) -> None:
        tree = self.summary_trees[key]
        for item in tree.get_children():
            tree.delete(item)
        for imei, total, defects, refined, filtered in summarize_imeis(self.current_time_filtered_patches(key)):
            tree.insert("", "end", values=(imei, total, defects, refined, filtered))

    def populate_face_rate(self, key: str) -> None:
        tree = self.face_rate_trees[key]
        for item in tree.get_children():
            tree.delete(item)

        denominator_imeis = self.current_rate_denominator_imeis(key)
        patches = self.current_imei_filtered_patches(key)
        for face, total, defects, defect_rate, refined, refined_rate, filtered, filtered_rate in summarize_face_presence(
            patches,
            denominator_imeis,
        ):
            tree.insert(
                "",
                "end",
                values=(
                    face,
                    format_presence_rate(defects, total, defect_rate),
                    format_presence_rate(refined, total, refined_rate),
                    format_presence_rate(filtered, total, filtered_rate),
                ),
            )

    def update_panel_status(self, key: str, patches: list[PatchImage], counts: dict[str, list[list[int]]]) -> str:
        face_totals = ", ".join(f"{face}={sum(sum(row) for row in counts[face])}" for face in FACES)
        shown_imei_count = len(self.current_rate_denominator_imeis(key))
        total_imei_count = len(self.panel_imeis[key])
        selected = self.date_options.get(self.date_vars[key].get(), "")
        details = [
            f"date={selected or '-'}",
            f"time={self.format_selected_time_range(key)}",
            f"inspections={self.panel_inspections[key]:,}",
            f"scanned patches={len(self.panel_patches[key]):,}",
            f"shown patches={len(patches):,}",
            f"imeis={shown_imei_count:,}/{total_imei_count:,}",
            f"max cell={max_count(counts):,}",
        ]
        if self.panel_missing_faces[key]:
            details.append(f"missing roots={','.join(self.panel_missing_faces[key])}")
        if self.panel_errors[key]:
            details.append(self.format_errors(self.panel_errors[key]))
        self.detail_vars[key].set(" | ".join(details) + "\n" + face_totals)
        return f"{key}: {selected or '-'} shown={len(patches):,}"

    def on_summary_select(self, key: str) -> None:
        tree = self.summary_trees[key]
        selected = tree.selection()
        if not selected:
            return
        values = tree.item(selected[0], "values")
        if not values:
            return
        self.imei_var.set(values[0])
        self.refresh_visuals()

    def on_cell_click(self, key: str, face: str, row: int, col: int) -> None:
        patches = [
            patch
            for patch in self.current_filtered_patches(key)
            if patch.face == face and patch.row == row and patch.col == col
        ]
        if not patches:
            return
        viewer = self.patch_viewers.get(key)
        if viewer is None or not viewer.winfo_exists():
            viewer = PatchViewerWindow(self)
            self.patch_viewers[key] = viewer
        selected = self.date_options.get(self.date_vars[key].get(), "-")
        viewer.show_patches(f"{PANEL_TITLES[key]} {selected} {face}[{row}][{col}]", patches)

    def current_filtered_patches(self, key: str) -> list[PatchImage]:
        category = self.category_var.get() or ALL_VALUE
        imei = self.imei_var.get() or ALL_VALUE
        return filter_patches(self.current_time_filtered_patches(key), category, imei)

    def current_time_range(self, key: str) -> tuple[str | None, str | None]:
        return normalize_time_range(
            parse_time_option(self.time_start_vars[key].get()),
            parse_time_option(self.time_end_vars[key].get()),
        )

    def current_time_filtered_patches(self, key: str) -> list[PatchImage]:
        start, end = self.current_time_range(key)
        return filter_patches_by_time(self.panel_patches[key], start, end)

    def current_time_filtered_inspection_records(self, key: str) -> tuple[InspectionMeta, ...]:
        start, end = self.current_time_range(key)
        return filter_inspection_records_by_time(self.panel_inspection_records[key], start, end)

    def current_imei_filtered_patches(self, key: str) -> list[PatchImage]:
        imei = self.imei_var.get() or ALL_VALUE
        patches = self.current_time_filtered_patches(key)
        if imei == ALL_VALUE:
            return patches
        return [patch for patch in patches if patch.imei == imei]

    def current_rate_denominator_imeis(self, key: str) -> set[str]:
        imei = self.imei_var.get() or ALL_VALUE
        time_filtered_imeis = {record.imei for record in self.current_time_filtered_inspection_records(key)}
        if not time_filtered_imeis:
            time_filtered_imeis = {patch.imei for patch in self.current_time_filtered_patches(key)}
        if imei != ALL_VALUE:
            return {imei} if imei in time_filtered_imeis else set()
        return time_filtered_imeis

    def format_selected_time_range(self, key: str) -> str:
        start, end = self.current_time_range(key)
        if start is None and end is None:
            return TIME_ALL_VALUE
        start_text = format_time_option(start) if start is not None else "Start"
        end_text = format_time_option(end) if end is not None else "End"
        return f"{start_text}~{end_text}"

    def _set_date_controls_enabled(self, enabled: bool) -> None:
        combo_state = "readonly" if enabled else "disabled"
        button_state = "normal" if enabled else "disabled"
        for combo in self.date_combos.values():
            combo.configure(state=combo_state)
        for button in self.apply_buttons.values():
            button.configure(state=button_state)

    def _set_filter_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.category_combo.configure(state=state)
        self.mode_combo.configure(state=state)
        self.imei_combo.configure(state=state)

    def format_errors(self, errors: Counter[str]) -> str:
        return "errors: " + ", ".join(f"{key}={value}" for key, value in sorted(errors.items()) if value)


def run_self_test() -> None:
    layout = compute_face_layout(1240, 780)
    assert round(layout["A"].width, 6) == round(layout["BT"].width, 6)
    assert round(layout["C"].width, 6) == round(layout["A"].width, 6)
    assert round(layout["A"].height, 6) == round(layout["BL"].height, 6)
    assert round(layout["C"].height, 6) == round(layout["A"].height, 6)
    assert round(layout["BR"].height, 6) == round(layout["A"].height, 6)

    assert parse_date_folder_name("260528") == date(2026, 5, 28)
    assert parse_date_folder_name("260230") is None
    assert load_saved_roots(Path("__missing_set_visualizer_config__.json")) == {}
    assert normalize_category_name("Defect") == "Defects"
    assert normalize_category_name("filtered") == "Filtered"
    inspection = parse_inspection_folder_name("122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW")
    assert inspection is not None
    assert inspection.hhmmss == "122850"
    assert inspection.imei == "ABC1234567890"
    assert inspection.model_file == "SM-S948-SMART_COSMETIC_V26.03.10.0"
    assert inspection.color_code == "ZW"

    patch = parse_patch_filename("[0][9][C_Center][0].png", "Defects")
    assert patch == PatchMeta(category="Defects", row=0, col=9, defect_type="C_Center", vector="0")
    assert to_display_cell("A", patch) == (9, 0)
    prefix_patch = parse_patch_filename("Filtered[1][1][B_Top][0].png")
    assert prefix_patch == PatchMeta(category="Filtered", row=1, col=1, defect_type="B_Top", vector="0")
    bl_patch = parse_patch_filename("[1][30][BLeft_Bottom][0].png", "Defects")
    assert bl_patch is not None
    assert face_from_patch_type("BL/BR", bl_patch.defect_type) == "BL"
    assert to_display_cell("BL", bl_patch) == (30, 1)
    br_patch = parse_patch_filename("[0][23][BRight_Top][0].png", "Defects")
    assert br_patch is not None
    assert face_from_patch_type("BL/BR", br_patch.defect_type) == "BR"
    assert to_display_cell("BR", br_patch) == (23, 0)
    bt_patch = parse_patch_filename("[20][1][BTop_Right][0].png", "Refined")
    assert bt_patch is not None
    assert face_from_patch_type("BT/BB", bt_patch.defect_type) == "BT"
    assert to_display_cell("BT", bt_patch) == (1, 20)
    bb_patch = parse_patch_filename("[12][0][BBottom_Left][0].png", "Filtered")
    assert bb_patch is not None
    assert face_from_patch_type("BT/BB", bb_patch.defect_type) == "BB"
    extended_refined = parse_patch_filename("[260528][ABC1234567890][0][23][BRight][Center][0].png", "Refined")
    assert extended_refined == PatchMeta(
        category="Refined",
        row=0,
        col=23,
        defect_type="BRight_Center",
        vector="0",
    )
    assert face_from_patch_type("BL/BR", extended_refined.defect_type) == "BR"
    assert to_display_cell("BR", extended_refined) == (23, 0)
    assert (
        build_patches_filename("260528", "ABC1234567890", "BR", extended_refined)
        == "[260528][ABC1234567890][0][23][BRight][Center][0].png"
    )
    original_ref = build_original_patch_ref(Path("inspection"), "260528", inspection, "BR", extended_refined)
    assert original_ref == PatchFileRef(
        category="Patches",
        row=0,
        col=23,
        defect_type="BRight_Center",
        vector="0",
        path=Path("inspection") / "Patches" / "[260528][ABC1234567890][0][23][BRight][Center][0].png",
    )
    errors: Counter[str] = Counter()
    built_patch = _build_patch_image(
        "BL/BR",
        "260528",
        Path("inspection"),
        inspection,
        Path("[260528][ABC1234567890][0][23][BRight][Center][0].png"),
        errors,
        "Refined",
    )
    assert built_patch is not None
    assert built_patch.face == "BR"
    assert built_patch.original_patch == original_ref
    assert not errors
    original_candidates = original_patch_path_candidates(built_patch)
    assert Path("inspection") / "Patches" / "[260528][ABC1234567890][0][23][BRight][Center][0].png" in original_candidates

    a_like_patch = PatchImage(
        "A",
        "260528",
        "122850",
        "ABC1234567890",
        "MODEL_A",
        "ZW",
        "Defects",
        9,
        0,
        "C_Center",
        "0",
        Path("inspection") / "Defects" / "[0][9][C_Center][0].png",
        build_original_patch_ref(
            Path("inspection"),
            "260528",
            inspection,
            "A",
            PatchMeta("Defects", 0, 9, "C_Center", "0"),
        ),
    )
    a_candidates = original_patch_path_candidates(a_like_patch)
    assert Path("inspection") / "Patches" / "[260528][ABC1234567890][0][9][A][Center][0].png" in a_candidates
    assert Path("inspection") / "Patches" / "[260528][ABC1234567890][0][9][C][Center][0].png" in a_candidates
    assert Path("inspection") / "Patches" / "[0][9][C_Center][0].png" in a_candidates
    extended_filtered = parse_patch_filename("[260528][ABC1234567890][20][1][BTop][Center][1].png", "Filtered")
    assert extended_filtered == PatchMeta(
        category="Filtered",
        row=20,
        col=1,
        defect_type="BTop_Center",
        vector="1",
    )
    assert face_from_patch_type("BT/BB", extended_filtered.defect_type) == "BT"
    assert to_display_cell("BT", extended_filtered) == (1, 20)
    swapped_a = parse_patch_filename("[30][19][C_Center][0].png", "Defects")
    assert swapped_a is not None
    assert to_display_cell("A", swapped_a) == (30, 19)
    assert face_from_patch_type("BL/BR", bt_patch.defect_type) is None
    assert to_display_cell("BL", PatchMeta("Defects", 2, 30, "B_Left", "0")) is None
    assert format_time_option("122850") == "12:28:50"
    assert parse_time_option("12:28:50") == "122850"
    assert parse_time_option(TIME_ALL_VALUE) is None
    assert normalize_time_range("130000", "120000") == ("120000", "130000")

    sample_patches = [
        PatchImage("A", "260528", "122850", "IMEI1", "MODEL_A", "ZW", "Defects", 0, 9, "C_Center", "0", Path("a")),
        PatchImage("A", "260528", "122850", "IMEI1", "MODEL_A", "ZW", "Refined", 0, 9, "C_Center", "0", Path("b")),
        PatchImage("C", "260528", "122851", "IMEI2", "MODEL_A", "ZW", "Defects", 30, 19, "C_Center", "0", Path("c")),
        PatchImage("BT", "260528", "122852", "IMEI3", "MODEL_A", "ZW", "Refined", 1, 20, "B_Top", "0", Path("d")),
    ]
    counts = aggregate_total_patches(filter_patches(sample_patches, ALL_VALUE, ALL_VALUE))
    assert counts["A"][0][9] == 2
    assert counts["C"][30][19] == 1
    assert counts["BT"][1][20] == 1
    unique_counts = aggregate_unique_imeis(sample_patches)
    assert unique_counts["A"][0][9] == 1
    summary = summarize_imeis(sample_patches)
    assert summary[0][0] == "IMEI1"
    face_presence = summarize_face_presence(sample_patches, {"IMEI1", "IMEI2", "IMEI3", "IMEI4"})
    by_face = {row[0]: row for row in face_presence}
    assert by_face["A"][1] == 4
    assert by_face["A"][2:8] == (1, 0.25, 1, 0.25, 0, 0.0)
    assert by_face["C"][2:8] == (1, 0.25, 0, 0.0, 0, 0.0)
    assert by_face["BT"][2:8] == (0, 0.0, 1, 0.25, 0, 0.0)
    assert format_presence_rate(1, 4, 0.25) == "1/4 (25.0%)"
    time_filtered = filter_patches_by_time(sample_patches, "122851", "122852")
    assert [patch.imei for patch in time_filtered] == ["IMEI2", "IMEI3"]
    reversed_time_filtered = filter_patches_by_time(sample_patches, "122852", "122851")
    assert [patch.imei for patch in reversed_time_filtered] == ["IMEI2", "IMEI3"]
    inspection_records = (
        InspectionMeta("122850", "IMEI1", "MODEL_A", "ZW"),
        InspectionMeta("122851", "IMEI2", "MODEL_A", "ZW"),
        InspectionMeta("122852", "IMEI3", "MODEL_A", "ZW"),
    )
    filtered_records = filter_inspection_records_by_time(inspection_records, "122851", None)
    assert [record.imei for record in filtered_records] == ["IMEI2", "IMEI3"]

    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize phone set patch counts from image folders.")
    parser.add_argument("--self-test", action="store_true", help="Run parser and aggregation checks without opening UI.")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    os.chdir(Path(__file__).resolve().parent)
    app = SetVisualizerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
