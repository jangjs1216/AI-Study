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
ALL_VALUE = "All"
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
CATEGORY_LOOKUP = {category.lower(): category for category in CATEGORIES}
DATE_FOLDER_RE = re.compile(r"^\d{6}$")
PATCH_RE = re.compile(
    r"^\[(?P<row>\d+)\]\[(?P<col>\d+)\]\[(?P<defect_type>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$",
    re.IGNORECASE,
)
PREFIX_PATCH_RE = re.compile(
    r"^(?P<category>[^\[]+)\[(?P<row>\d+)\]\[(?P<col>\d+)\]\[(?P<defect_type>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$",
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


@dataclass(frozen=True)
class ScanResult:
    patches: list[PatchImage]
    inspections: int
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
    max_file_x, max_file_y = FILE_COORD_LIMITS[face]
    if not (0 <= meta.row <= max_file_x and 0 <= meta.col <= max_file_y):
        return None

    display_row = meta.col
    display_col = meta.row

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
            except OSError:
                errors["scan worker os error"] += 1
                continue
            except Exception:
                errors["scan worker error"] += 1
                continue
            patches.extend(worker_patches)
            errors.update(worker_errors)

    return ScanResult(
        patches=patches,
        inspections=inspections,
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

    for category in CATEGORIES:
        category_dir = inspection_dir / category
        if not category_dir.exists():
            continue
        if not category_dir.is_dir():
            errors["category path is not folder"] += 1
            continue

        try:
            category_files = [path for path in category_dir.iterdir() if path.is_file()]
        except OSError:
            errors[f"{category}: read error"] += 1
            continue

        for path in category_files:
            patch = _build_patch_image(root_slot, day_folder, inspection, path, errors, category)
            if patch is not None:
                patches.append(patch)

    try:
        direct_files = [path for path in inspection_dir.iterdir() if path.is_file()]
    except OSError:
        errors["inspection direct read error"] += 1
        direct_files = []

    for path in direct_files:
        patch = _build_patch_image(root_slot, day_folder, inspection, path, errors, None)
        if patch is not None:
            patches.append(patch)

    return patches, errors


def _build_patch_image(
    root_slot: str,
    day_folder: str,
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
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.counts = make_empty_counts()
        self.max_value = 0
        self.canvas = tk.Canvas(self, bg="#f1f5f9", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

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


class SetVisualizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Set Visualizer")
        self.geometry("1720x900")
        self.minsize(1280, 760)

        self.root_vars = {root_slot: tk.StringVar() for root_slot in ROOT_SLOTS}
        self.date_vars = {key: tk.StringVar() for key in PANEL_KEYS}
        self.category_var = tk.StringVar(value=ALL_VALUE)
        self.mode_var = tk.StringVar(value=MODE_TOTAL_PATCHES)
        self.imei_var = tk.StringVar(value=ALL_VALUE)
        self.status_var = tk.StringVar(value="Ready")
        self.detail_vars = {key: tk.StringVar(value="No date loaded") for key in PANEL_KEYS}
        self.date_options: dict[str, str] = {}
        self.panel_patches: dict[str, list[PatchImage]] = {key: [] for key in PANEL_KEYS}
        self.panel_errors: dict[str, Counter[str]] = {key: Counter() for key in PANEL_KEYS}
        self.panel_missing_faces: dict[str, tuple[str, ...]] = {key: () for key in PANEL_KEYS}
        self.panel_inspections: dict[str, int] = {key: 0 for key in PANEL_KEYS}
        self.date_combos: dict[str, ttk.Combobox] = {}
        self.apply_buttons: dict[str, ttk.Button] = {}
        self.canvases: dict[str, PhoneSetCanvas] = {}
        self.summary_trees: dict[str, ttk.Treeview] = {}
        self.scan_cache: dict[tuple[str, tuple[tuple[str, str], ...]], ScanResult] = {}
        self.pending_scan_panels: dict[tuple[str, tuple[tuple[str, str], ...]], set[str]] = {}
        self.panel_scan_keys: dict[str, tuple[str, tuple[tuple[str, str], ...]] | None] = {
            key: None for key in PANEL_KEYS
        }

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

        canvas = PhoneSetCanvas(panel)
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
            result = ScanResult(patches=[], inspections=0, errors=Counter({"scan failed": 1}), missing_faces=())
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
            self.panel_errors[panel] = result.errors
            self.panel_missing_faces[panel] = result.missing_faces
            self.panel_inspections[panel] = result.inspections
        if panels:
            self.populate_imei_choices()
            self._set_filter_controls_enabled(True)
            self.refresh_visuals()

    def populate_imei_choices(self) -> None:
        current = self.imei_var.get()
        imeis = sorted({patch.imei for patches in self.panel_patches.values() for patch in patches})
        values = (ALL_VALUE, *imeis)
        self.imei_combo.configure(values=values)
        if current in values:
            self.imei_var.set(current)
        else:
            self.imei_var.set(ALL_VALUE)

    def refresh_visuals(self) -> None:
        category = self.category_var.get() or ALL_VALUE
        imei = self.imei_var.get() or ALL_VALUE

        panel_parts: list[str] = []
        for key in PANEL_KEYS:
            patches = filter_patches(self.panel_patches[key], category, imei)
            if self.mode_var.get() == MODE_UNIQUE_IMEI:
                counts = aggregate_unique_imeis(patches)
            else:
                counts = aggregate_total_patches(patches)
            self.canvases[key].set_counts(counts)
            self.populate_summary(key)
            panel_parts.append(self.update_panel_status(key, patches, counts))

        self.status_var.set(
            f"category={category} | mode={self.mode_var.get()} | imei={imei} | " + " | ".join(panel_parts)
        )

    def populate_summary(self, key: str) -> None:
        tree = self.summary_trees[key]
        for item in tree.get_children():
            tree.delete(item)
        for imei, total, defects, refined, filtered in summarize_imeis(self.panel_patches[key]):
            tree.insert("", "end", values=(imei, total, defects, refined, filtered))

    def update_panel_status(self, key: str, patches: list[PatchImage], counts: dict[str, list[list[int]]]) -> str:
        face_totals = ", ".join(f"{face}={sum(sum(row) for row in counts[face])}" for face in FACES)
        imei_count = len({patch.imei for patch in self.panel_patches[key]})
        selected = self.date_options.get(self.date_vars[key].get(), "")
        details = [
            f"date={selected or '-'}",
            f"inspections={self.panel_inspections[key]:,}",
            f"scanned patches={len(self.panel_patches[key]):,}",
            f"shown patches={len(patches):,}",
            f"imeis={imei_count:,}",
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
    assert face_from_patch_type("BL/BR", bt_patch.defect_type) is None
    assert to_display_cell("BL", PatchMeta("Defects", 2, 30, "B_Left", "0")) is None

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
