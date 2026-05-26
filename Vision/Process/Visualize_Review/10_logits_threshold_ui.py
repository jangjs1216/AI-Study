# -*- coding: utf-8 -*-
"""Interactive threshold explorer for class-1 jet logits manifests.

Expected CSV columns:
    group, file_name, original_path, logits_class0_path, logits_class1_path,
    mask_path, overlap_class0_original_path, overlap_class0_overlap_path,
    overlap_class1_original_path, overlap_class1_overlap_path

Run:
    python Vision/Process/Visualize_Review/10_logits_threshold_ui.py --csv path/to/manifest.csv

CSV open only loads metadata. Press Reload after setting Samples/group to preprocess a
random sample from each group.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError("Pillow is required. Install it with: python -m pip install pillow") from exc


REQUIRED_COLUMNS = (
    "group",
    "file_name",
    "original_path",
    "logits_class0_path",
    "logits_class1_path",
    "mask_path",
    "overlap_class0_original_path",
    "overlap_class0_overlap_path",
    "overlap_class1_original_path",
    "overlap_class1_overlap_path",
)

PATH_COLUMNS = (
    "original_path",
    "logits_class0_path",
    "logits_class1_path",
    "mask_path",
    "overlap_class0_original_path",
    "overlap_class0_overlap_path",
    "overlap_class1_original_path",
    "overlap_class1_overlap_path",
)

ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
PREVIEW_SIZE = 360
MISSING = "<missing>"

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - Pillow < 9 compatibility
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass
class CsvData:
    headers: list[str]
    rows: list[dict[str, str]]
    encoding: str


@dataclass
class ImageRecord:
    index: int
    group: str
    file_name: str
    paths: dict[str, Path | None]
    hist: np.ndarray | None = None
    survival: np.ndarray | None = None
    max_bin: int = -1
    load_error: str = ""

    def alive_pixels(self, threshold_bin: int) -> int:
        if self.survival is None or threshold_bin < 0:
            return 0
        threshold_bin = min(max(threshold_bin, 0), 255)
        return int(self.survival[threshold_bin])

    @property
    def total_pixels(self) -> int:
        if self.hist is None:
            return 0
        return int(self.hist.sum())

    @property
    def max_score(self) -> float:
        if self.max_bin < 0:
            return float("nan")
        return self.max_bin / 255.0


def read_csv_flexible(path: Path) -> CsvData:
    errors: list[str] = []
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                sample = file.read(8192)
                file.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                except csv.Error:
                    dialect = csv.excel

                reader = csv.DictReader(file, dialect=dialect)
                if not reader.fieldnames:
                    raise ValueError("CSV header was not found.")

                headers = [header.strip() for header in reader.fieldnames if header is not None]
                rows: list[dict[str, str]] = []
                for row in reader:
                    clean = {
                        (key.strip() if key is not None else ""): (value.strip() if value is not None else "")
                        for key, value in row.items()
                    }
                    rows.append(clean)
                return CsvData(headers=headers, rows=rows, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface attempted encodings to the UI
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError("Could not read CSV.\n" + "\n".join(errors))


def validate_columns(headers: Iterable[str]) -> None:
    header_set = set(headers)
    missing = [column for column in REQUIRED_COLUMNS if column not in header_set]
    if missing:
        raise ValueError("Missing required CSV columns:\n" + "\n".join(missing))


def resolve_path(raw_value: str, csv_path: Path) -> Path | None:
    text = str(raw_value).strip()
    if not text:
        return None

    path = Path(text)
    candidates = [path] if path.is_absolute() else [csv_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.resolve()
            except OSError:
                return candidate
    return candidates[0]


def build_records(csv_path: Path, rows: list[dict[str, str]]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for index, row in enumerate(rows):
        group = row.get("group", "").strip() or MISSING
        file_name = row.get("file_name", "").strip()
        if not file_name:
            original = row.get("original_path", "").strip()
            file_name = Path(original).name if original else f"row_{index}"

        paths = {column: resolve_path(row.get(column, ""), csv_path) for column in PATH_COLUMNS}
        records.append(ImageRecord(index=index, group=group, file_name=file_name, paths=paths))
    return records


@lru_cache(maxsize=4)
def matplotlib_jet_lut() -> np.ndarray:
    try:
        import matplotlib.cm as cm  # type: ignore

        cmap = cm.get_cmap("jet", 256)
        lut = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
        return np.clip(np.rint(lut * 255.0), 0, 255).astype(np.uint8)
    except Exception:  # noqa: BLE001 - keep UI usable without matplotlib
        return fallback_jet_lut()


@lru_cache(maxsize=4)
def opencv_jet_lut() -> np.ndarray:
    try:
        import cv2  # type: ignore

        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 3)
        return bgr[:, ::-1].copy()
    except Exception:  # noqa: BLE001 - OpenCV is optional
        return fallback_jet_lut()


def fallback_jet_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.rint(np.stack([red, green, blue], axis=1) * 255.0).astype(np.uint8)


def nearest_lut_bins(colors: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, float]:
    if colors.size == 0:
        return np.zeros((0,), dtype=np.uint8), float("inf")

    colors_i32 = colors.astype(np.int32, copy=False)
    lut_i32 = lut.astype(np.int32, copy=False)
    bins = np.empty((len(colors_i32),), dtype=np.uint8)
    total_distance = 0.0
    chunk_size = 4096

    for start in range(0, len(colors_i32), chunk_size):
        chunk = colors_i32[start : start + chunk_size]
        diff = chunk[:, None, :] - lut_i32[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        argmin = np.argmin(distances, axis=1).astype(np.uint8)
        bins[start : start + len(chunk)] = argmin
        total_distance += float(np.min(distances, axis=1).sum())

    return bins, total_distance / max(1, len(colors_i32))


def decode_rgb_to_score_bins(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "luminance":
        gray = np.rint(
            0.299 * rgb[..., 0].astype(np.float32)
            + 0.587 * rgb[..., 1].astype(np.float32)
            + 0.114 * rgb[..., 2].astype(np.float32)
        )
        return np.clip(gray, 0, 255).astype(np.uint8)

    flat = rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    if mode == "opencv_jet":
        bins, _distance = nearest_lut_bins(unique_colors, opencv_jet_lut())
    elif mode == "matplotlib_jet":
        bins, _distance = nearest_lut_bins(unique_colors, matplotlib_jet_lut())
    else:
        mpl_bins, mpl_distance = nearest_lut_bins(unique_colors, matplotlib_jet_lut())
        cv_bins, cv_distance = nearest_lut_bins(unique_colors, opencv_jet_lut())
        bins = cv_bins if cv_distance < mpl_distance else mpl_bins

    return bins[inverse].reshape(rgb.shape[:2]).astype(np.uint8)


@lru_cache(maxsize=16)
def load_rgb_image(path_text: str) -> Image.Image:
    with Image.open(path_text) as image:
        return image.convert("RGB").copy()


@lru_cache(maxsize=8)
def load_score_map(path_text: str, mode: str) -> np.ndarray:
    image = load_rgb_image(path_text)
    rgb = np.asarray(image, dtype=np.uint8)
    return decode_rgb_to_score_bins(rgb, mode)


def histogram_from_score_map(score_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    hist = np.bincount(score_map.ravel(), minlength=256).astype(np.int64)
    survival = np.cumsum(hist[::-1], dtype=np.int64)[::-1]
    nonzero_bins = np.flatnonzero(hist)
    max_bin = int(nonzero_bins[-1]) if len(nonzero_bins) else -1
    return hist, survival, max_bin


def threshold_to_bin(threshold: float) -> int:
    value = int(math.ceil(float(threshold) * 255.0 - 1e-9))
    return min(max(value, 0), 255)


def score_text(score: float) -> str:
    if not np.isfinite(score):
        return "nan"
    return f"{score:.3f}"


def fit_image(image: Image.Image, size: int = PREVIEW_SIZE) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def placeholder_image(title: str, detail: str = "", size: int = PREVIEW_SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = [title]
    if detail:
        lines.extend(wrap_text(detail, 46)[:6])
    y = 24
    for line in lines:
        draw.text((18, y), line, fill=(55, 55, 55), font=font)
        y += 18
    return image


def wrap_text(text: str, width: int) -> list[str]:
    words = text.replace("\\", "/").split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def make_mask_image(score_map: np.ndarray, threshold_bin: int) -> Image.Image:
    mask = (score_map >= threshold_bin).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def make_overlay_image(original: Image.Image, score_map: np.ndarray, threshold_bin: int) -> Image.Image:
    base = original.convert("RGBA")
    mask = score_map >= threshold_bin
    overlay = np.zeros((score_map.shape[0], score_map.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = np.where(mask, 140, 0).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGBA")
    if overlay_image.size != base.size:
        overlay_image = overlay_image.resize(base.size, RESAMPLE_NEAREST)
    return Image.alpha_composite(base, overlay_image).convert("RGB")


class LogitsThresholdApp:
    def __init__(self, root: tk.Tk, csv_path: Path | None = None) -> None:
        self.root = root
        self.root.title("Class-1 Jet Logits Threshold Explorer")
        self.root.geometry("1560x980")
        self.csv_path: Path | None = None
        self.all_records: list[ImageRecord] = []
        self.records: list[ImageRecord] = []
        self.selected_group: str | None = None
        self.selected_record_index: int | None = None
        self.group_rows: dict[str, str] = {}
        self.record_rows: dict[str, int] = {}
        self.photos: dict[str, ImageTk.PhotoImage] = {}

        self.threshold_var = tk.DoubleVar(value=0.50)
        self.threshold_text_var = tk.StringVar(value="0.500")
        self.min_pixels_var = tk.IntVar(value=1)
        self.samples_per_group_var = tk.IntVar(value=20)
        self.sample_seed_var = tk.IntVar(value=0)
        self.decode_mode_var = tk.StringVar(value="auto")
        self.show_all_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Open a manifest CSV.")
        self.csv_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")

        self._build_ui()
        if csv_path is not None:
            self.root.after(100, lambda: self.load_csv(csv_path))

    def _build_ui(self) -> None:
        control = ttk.Frame(self.root, padding=(10, 8))
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control, text="Open CSV", command=self.ask_open_csv).pack(side=tk.LEFT)
        ttk.Label(control, textvariable=self.csv_var, width=72, anchor="w").pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(control, text="Decode").pack(side=tk.LEFT, padx=(0, 5))
        mode_combo = ttk.Combobox(
            control,
            textvariable=self.decode_mode_var,
            width=15,
            state="readonly",
            values=("auto", "matplotlib_jet", "opencv_jet", "luminance"),
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 14))
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.recompute_histograms())

        ttk.Label(control, text="Samples/group").pack(side=tk.LEFT, padx=(0, 5))
        samples_per_group = ttk.Spinbox(
            control,
            textvariable=self.samples_per_group_var,
            from_=1,
            to=100_000,
            increment=1,
            width=7,
        )
        samples_per_group.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(control, text="Seed").pack(side=tk.LEFT, padx=(0, 5))
        sample_seed = ttk.Spinbox(
            control,
            textvariable=self.sample_seed_var,
            from_=0,
            to=1_000_000,
            increment=1,
            width=7,
        )
        sample_seed.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(control, text="Min pixels").pack(side=tk.LEFT, padx=(0, 5))
        min_pixels = ttk.Spinbox(
            control,
            textvariable=self.min_pixels_var,
            from_=1,
            to=1_000_000,
            increment=1,
            width=8,
            command=self.refresh_threshold_view,
        )
        min_pixels.pack(side=tk.LEFT, padx=(0, 14))
        min_pixels.bind("<KeyRelease>", lambda _event: self.refresh_threshold_view())

        ttk.Checkbutton(
            control,
            text="Show all sampled",
            variable=self.show_all_var,
            command=self.refresh_image_list,
        ).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Button(control, text="Reload", command=self.reload_sample).pack(side=tk.LEFT)

        threshold_frame = ttk.Frame(self.root, padding=(10, 0))
        threshold_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(threshold_frame, text="Threshold").pack(side=tk.LEFT, padx=(0, 8))
        scale = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            length=620,
            command=self.on_threshold_slider,
            showvalue=False,
        )
        scale.pack(side=tk.LEFT)
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_text_var, width=8)
        threshold_entry.pack(side=tk.LEFT, padx=(8, 18))
        threshold_entry.bind("<Return>", lambda _event: self.on_threshold_entry())
        threshold_entry.bind("<FocusOut>", lambda _event: self.on_threshold_entry())
        ttk.Label(threshold_frame, textvariable=self.summary_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        group_frame = ttk.LabelFrame(body, text="Groups", padding=(6, 6))
        body.add(group_frame, weight=1)

        group_columns = ("group", "alive", "sampled", "total", "pct", "pixels")
        self.group_tree = ttk.Treeview(group_frame, columns=group_columns, show="headings", height=22)
        for column, width in zip(group_columns, (210, 70, 70, 70, 70, 110)):
            self.group_tree.heading(column, text=column)
            self.group_tree.column(column, width=width, minwidth=50, anchor="center")
        self.group_tree.column("group", anchor="w")
        self.group_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        group_scroll = ttk.Scrollbar(group_frame, orient=tk.VERTICAL, command=self.group_tree.yview)
        group_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.group_tree.configure(yscrollcommand=group_scroll.set)
        self.group_tree.bind("<<TreeviewSelect>>", self.on_group_select)

        list_frame = ttk.LabelFrame(body, text="Images", padding=(6, 6))
        body.add(list_frame, weight=2)

        image_columns = ("file_name", "max", "pixels", "ratio", "error")
        self.image_tree = ttk.Treeview(list_frame, columns=image_columns, show="headings", height=22)
        for column, width in zip(image_columns, (260, 70, 90, 70, 170)):
            self.image_tree.heading(column, text=column)
            self.image_tree.column(column, width=width, minwidth=50, anchor="center")
        self.image_tree.column("file_name", anchor="w")
        self.image_tree.column("error", anchor="w")
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_tree.yview)
        image_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=image_scroll.set)
        self.image_tree.bind("<<TreeviewSelect>>", self.on_image_select)

        viewer_frame = ttk.LabelFrame(body, text="Preview", padding=(8, 8))
        body.add(viewer_frame, weight=4)

        self.meta_label = ttk.Label(viewer_frame, text="", justify=tk.LEFT, anchor="w")
        self.meta_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        grid = ttk.Frame(viewer_frame)
        grid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.panel_labels: dict[str, ttk.Label] = {}
        panels = (
            ("original", "Original"),
            ("logits", "Class1 jet"),
            ("mask", "Threshold mask"),
            ("overlay", "Original + mask"),
        )
        for idx, (key, title) in enumerate(panels):
            panel = ttk.Frame(grid)
            panel.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=6, pady=6)
            ttk.Label(panel, text=title, anchor="center").pack(side=tk.TOP, fill=tk.X)
            label = ttk.Label(panel)
            label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.panel_labels[key] = label

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.show_blank_preview()

    def ask_open_csv(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Open manifest CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if path_text:
            self.load_csv(Path(path_text))

    def reload_sample(self) -> None:
        if self.csv_path is None:
            self.ask_open_csv()
            return
        self.prepare_sample_and_decode()

    def load_csv(self, path: Path) -> None:
        try:
            csv_data = read_csv_flexible(path)
            validate_columns(csv_data.headers)
            records = build_records(path, csv_data.rows)
        except Exception as exc:  # noqa: BLE001 - show clear UI error
            messagebox.showerror("CSV load failed", str(exc))
            self.status_var.set(f"CSV load failed: {exc}")
            return

        self.csv_path = path
        self.all_records = records
        self.records = []
        self.selected_group = None
        self.selected_record_index = None
        self.csv_var.set(str(path))
        self.populate_group_tree()

        groups = sorted({record.group for record in self.all_records})
        self.selected_group = groups[0] if groups else None
        if self.selected_group is not None:
            self.select_group(self.selected_group)

        self.summary_var.set(f"source rows={len(records)}, groups={len(groups)}. Press Reload to preprocess samples.")
        self.status_var.set(
            f"Loaded {len(records)} rows with {csv_data.encoding}. "
            "No heatmaps decoded yet; set Samples/group and press Reload."
        )
        self.show_blank_preview("Press Reload to preprocess sampled rows.")

    def current_samples_per_group(self) -> int:
        try:
            return max(1, int(self.samples_per_group_var.get()))
        except Exception:  # noqa: BLE001
            return 20

    def current_sample_seed(self) -> int:
        try:
            return int(self.sample_seed_var.get())
        except Exception:  # noqa: BLE001
            return 0

    def source_group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.all_records:
            counts[record.group] = counts.get(record.group, 0) + 1
        return counts

    def prepare_sample_and_decode(self) -> None:
        if not self.all_records:
            self.status_var.set("No CSV rows loaded.")
            return

        sample_count = self.current_samples_per_group()
        rng = random.Random(self.current_sample_seed())
        by_group: dict[str, list[ImageRecord]] = {}
        for record in self.all_records:
            by_group.setdefault(record.group, []).append(record)

        sampled: list[ImageRecord] = []
        for group in sorted(by_group):
            group_records = by_group[group]
            if len(group_records) <= sample_count:
                selected = list(group_records)
            else:
                selected = rng.sample(group_records, sample_count)
                selected.sort(key=lambda record: (record.file_name, record.index))
            sampled.extend(selected)

        self.records = sampled
        self.selected_record_index = None
        self.populate_group_tree()
        if self.selected_group not in by_group:
            self.selected_group = sorted(by_group)[0]
        self.status_var.set(
            f"Sampled {len(self.records)}/{len(self.all_records)} rows "
            f"({sample_count}/group, seed={self.current_sample_seed()}). Decoding heatmaps..."
        )
        self.root.update_idletasks()
        self.recompute_histograms(show_errors=False)

    def recompute_histograms(self, show_errors: bool = True) -> None:
        if not self.records:
            self.refresh_threshold_view()
            return

        load_score_map.cache_clear()
        load_rgb_image.cache_clear()
        mode = self.decode_mode_var.get()
        total = len(self.records)
        errors = 0

        for offset, record in enumerate(self.records, start=1):
            record.hist = None
            record.survival = None
            record.max_bin = -1
            record.load_error = ""
            logits_path = record.paths.get("logits_class1_path")
            if logits_path is None:
                record.load_error = "empty logits path"
                errors += 1
                continue
            if not logits_path.exists():
                record.load_error = "missing logits file"
                errors += 1
                continue

            try:
                score_map = load_score_map(str(logits_path), mode)
                record.hist, record.survival, record.max_bin = histogram_from_score_map(score_map)
            except Exception as exc:  # noqa: BLE001 - row-level error should not kill the app
                record.load_error = str(exc)
                errors += 1

            if offset == 1 or offset == total or offset % 25 == 0:
                self.status_var.set(f"Decoded {offset}/{total} heatmaps with mode={mode}...")
                self.root.update_idletasks()

        groups = sorted({record.group for record in self.all_records}) or sorted({record.group for record in self.records})
        if self.selected_group not in groups:
            self.selected_group = groups[0] if groups else None
        self.populate_group_tree()
        self.refresh_threshold_view()
        if self.selected_group is not None:
            self.select_group(self.selected_group)

        message = (
            f"Ready. sampled_rows={total}/{len(self.all_records) or total}, "
            f"groups={len(groups)}, decode_mode={mode}, errors={errors}"
        )
        self.status_var.set(message)
        if errors and show_errors:
            messagebox.showwarning("Heatmap decode warnings", f"{errors} rows could not be decoded. See image list.")

    def on_threshold_slider(self, raw_value: str) -> None:
        threshold = float(raw_value)
        self.threshold_text_var.set(f"{threshold:.3f}")
        self.refresh_threshold_view()

    def on_threshold_entry(self) -> None:
        try:
            threshold = float(self.threshold_text_var.get())
        except ValueError:
            self.threshold_text_var.set(f"{self.threshold_var.get():.3f}")
            return
        threshold = min(max(threshold, 0.0), 1.0)
        self.threshold_var.set(threshold)
        self.threshold_text_var.set(f"{threshold:.3f}")
        self.refresh_threshold_view()

    def current_threshold_bin(self) -> int:
        return threshold_to_bin(self.threshold_var.get())

    def current_min_pixels(self) -> int:
        try:
            return max(1, int(self.min_pixels_var.get()))
        except Exception:  # noqa: BLE001
            return 1

    def record_survives(self, record: ImageRecord, threshold_bin: int, min_pixels: int) -> bool:
        return record.alive_pixels(threshold_bin) >= min_pixels

    def populate_group_tree(self) -> None:
        for item in self.group_tree.get_children():
            self.group_tree.delete(item)
        self.group_rows.clear()

        source_counts = self.source_group_counts()
        groups = sorted(source_counts) if source_counts else sorted({record.group for record in self.records})
        for group in groups:
            item_id = self.group_tree.insert(
                "",
                tk.END,
                values=(group, 0, 0, source_counts.get(group, 0), "0.0%", 0),
            )
            self.group_rows[group] = item_id

    def refresh_threshold_view(self) -> None:
        threshold_bin = self.current_threshold_bin()
        min_pixels = self.current_min_pixels()
        total_alive = 0
        sampled_rows = 0
        source_rows = len(self.all_records) if self.all_records else len(self.records)
        total_pixels = 0

        source_counts = self.source_group_counts()
        group_stats: dict[str, dict[str, int]] = {
            group: {"alive": 0, "sampled": 0, "total": total, "pixels": 0}
            for group, total in source_counts.items()
        }
        for record in self.records:
            stats = group_stats.setdefault(record.group, {"alive": 0, "sampled": 0, "total": 0, "pixels": 0})
            stats["sampled"] += 1
            alive_pixels = record.alive_pixels(threshold_bin)
            stats["pixels"] += alive_pixels
            if alive_pixels >= min_pixels:
                stats["alive"] += 1
            sampled_rows += 1
            total_pixels += alive_pixels

        for group, stats in group_stats.items():
            item_id = self.group_rows.get(group)
            if item_id is None:
                continue
            pct = (stats["alive"] / stats["sampled"] * 100.0) if stats["sampled"] else 0.0
            self.group_tree.item(
                item_id,
                values=(
                    group,
                    stats["alive"],
                    stats["sampled"],
                    stats["total"],
                    f"{pct:.1f}%",
                    stats["pixels"],
                ),
            )
            total_alive += stats["alive"]

        threshold = self.threshold_var.get()
        self.summary_var.set(
            f"score>={threshold:.3f} (bin {threshold_bin}/255), "
            f"alive sampled images={total_alive}/{sampled_rows}, source rows={source_rows}, alive pixels={total_pixels}"
        )
        self.refresh_image_list()
        self.render_selected_record()

    def refresh_image_list(self) -> None:
        for item in self.image_tree.get_children():
            self.image_tree.delete(item)
        self.record_rows.clear()

        if self.selected_group is None:
            return

        threshold_bin = self.current_threshold_bin()
        min_pixels = self.current_min_pixels()
        show_all = self.show_all_var.get()
        records = [record for record in self.records if record.group == self.selected_group]
        rows: list[tuple[int, ImageRecord, int]] = []

        for record in records:
            alive_pixels = record.alive_pixels(threshold_bin)
            if show_all or alive_pixels >= min_pixels:
                rows.append((alive_pixels, record, record.total_pixels))

        rows.sort(key=lambda item: (item[0], item[1].max_bin, item[1].file_name), reverse=True)
        for alive_pixels, record, total_pixels in rows:
            ratio = alive_pixels / total_pixels * 100.0 if total_pixels else 0.0
            values = (
                record.file_name,
                score_text(record.max_score),
                alive_pixels,
                f"{ratio:.2f}%",
                record.load_error,
            )
            item_id = self.image_tree.insert("", tk.END, values=values)
            self.record_rows[item_id] = record.index

        if self.selected_record_index is None and rows:
            self.selected_record_index = rows[0][1].index
            self.select_record(self.selected_record_index)

    def on_group_select(self, _event: object) -> None:
        selection = self.group_tree.selection()
        if not selection:
            return
        values = self.group_tree.item(selection[0], "values")
        if not values:
            return
        self.selected_group = str(values[0])
        self.selected_record_index = None
        self.refresh_image_list()
        if self.image_tree.get_children():
            first_item = self.image_tree.get_children()[0]
            self.image_tree.selection_set(first_item)
            self.image_tree.focus(first_item)
            self.on_image_select(None)
        elif not self.records:
            self.show_blank_preview("Press Reload to preprocess sampled rows.")
        else:
            self.show_blank_preview(f"No rows survive in group: {self.selected_group}")

    def select_group(self, group: str) -> None:
        item_id = self.group_rows.get(group)
        if item_id is None:
            return
        self.group_tree.selection_set(item_id)
        self.group_tree.focus(item_id)
        self.group_tree.see(item_id)
        self.on_group_select(None)

    def on_image_select(self, _event: object) -> None:
        selection = self.image_tree.selection()
        if not selection:
            return
        record_index = self.record_rows.get(selection[0])
        if record_index is None:
            return
        self.selected_record_index = record_index
        self.render_selected_record()

    def select_record(self, record_index: int) -> None:
        for item_id, item_record_index in self.record_rows.items():
            if item_record_index == record_index:
                self.image_tree.selection_set(item_id)
                self.image_tree.focus(item_id)
                self.image_tree.see(item_id)
                break

    def selected_record(self) -> ImageRecord | None:
        if self.selected_record_index is None:
            return None
        for record in self.records:
            if record.index == self.selected_record_index:
                return record
        return None

    def render_selected_record(self) -> None:
        record = self.selected_record()
        if record is None:
            self.show_blank_preview()
            return

        threshold_bin = self.current_threshold_bin()
        alive_pixels = record.alive_pixels(threshold_bin)
        total_pixels = record.total_pixels
        ratio = alive_pixels / total_pixels * 100.0 if total_pixels else 0.0
        logits_path = record.paths.get("logits_class1_path")
        original_path = record.paths.get("original_path")

        self.meta_label.configure(
            text=(
                f"group={record.group}\n"
                f"file={record.file_name}\n"
                f"max={score_text(record.max_score)}, alive_pixels={alive_pixels}/{total_pixels} ({ratio:.2f}%), "
                f"threshold={self.threshold_var.get():.3f}, min_pixels={self.current_min_pixels()}"
            )
        )

        original_image = self.safe_load_image(original_path, "Original")
        logits_image = self.safe_load_image(logits_path, "Class1 jet")

        score_map: np.ndarray | None = None
        if logits_path is not None and logits_path.exists():
            try:
                score_map = load_score_map(str(logits_path), self.decode_mode_var.get())
            except Exception as exc:  # noqa: BLE001
                score_map = None
                self.status_var.set(f"Preview decode failed: {exc}")

        if score_map is None:
            mask_image = placeholder_image("No score map", record.load_error)
            overlay_image = placeholder_image("No overlay", record.load_error)
        else:
            mask_image = make_mask_image(score_map, threshold_bin)
            overlay_base = original_image if original_image is not None else logits_image
            if overlay_base is None:
                overlay_image = placeholder_image("No overlay base")
            else:
                overlay_image = make_overlay_image(overlay_base, score_map, threshold_bin)

        self.set_panel_image("original", original_image or placeholder_image("Original unavailable"))
        self.set_panel_image("logits", logits_image or placeholder_image("Class1 jet unavailable"))
        self.set_panel_image("mask", mask_image)
        self.set_panel_image("overlay", overlay_image)

    def safe_load_image(self, path: Path | None, title: str) -> Image.Image | None:
        if path is None:
            return None
        if not path.exists():
            return placeholder_image(f"{title} missing", str(path))
        try:
            return load_rgb_image(str(path))
        except Exception as exc:  # noqa: BLE001
            return placeholder_image(f"{title} load failed", str(exc))

    def set_panel_image(self, key: str, image: Image.Image) -> None:
        photo = ImageTk.PhotoImage(fit_image(image))
        self.photos[key] = photo
        self.panel_labels[key].configure(image=photo)

    def show_blank_preview(self, detail: str = "") -> None:
        image = placeholder_image("No selection", detail)
        for key in self.panel_labels:
            self.set_panel_image(key, image)
        self.meta_label.configure(text=detail)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore class-1 jet logits thresholds by group.")
    parser.add_argument("--csv", type=Path, default=None, help="Manifest CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = tk.Tk()
    app = LogitsThresholdApp(root, csv_path=args.csv)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
