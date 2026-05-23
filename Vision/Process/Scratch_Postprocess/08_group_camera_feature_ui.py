# -*- coding: utf-8 -*-
"""Interactive group/camera feature explorer for scratch component CSV files.

Run:
    python Vision/Process/Scratch_Postprocess/08_group_camera_feature_ui.py

Optional:
    python Vision/Process/Scratch_Postprocess/08_group_camera_feature_ui.py --csv path/to/features.csv --image-root path/to/images
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageFilter, ImageTk
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError("Pillow가 필요합니다. `python -m pip install pillow` 후 다시 실행하세요.") from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib import font_manager  # noqa: E402


TRUE_DEFECT_KEYWORD = "진불스크래치"
MICRO_SCRATCH_KEYWORD = "미세스크래치"
CAMERA_PATTERN = re.compile(r"\[([^\[\]]+)\](?=\.[^.]+$)")

ID_LIKE_COLUMNS = {
    "image_path",
    "mask_path",
    "mask_raw_path",
    "component_id",
    "group",
    "camera_mode",
    "defect_type",
    "row_id",
    "sample_id",
    "label",
}

DEFECT_COLORS = {
    "진불스크래치": "#d62728",
    "미세스크래치": "#ff7f0e",
    "기타": "#4c78a8",
}

KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",
    "맑은 고딕",
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "AppleGothic",
    "Arial Unicode MS",
]


def configure_korean_matplotlib_font() -> str | None:
    """Configure matplotlib to render Korean labels on common OS setups."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in KOREAN_FONT_CANDIDATES if name in available), None)

    if selected is None and sys.platform.startswith("win"):
        for font_path in [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
            Path("C:/Windows/Fonts/NanumGothic.ttf"),
        ]:
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                selected = font_manager.FontProperties(fname=str(font_path)).get_name()
                break

    if selected:
        matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected


KOREAN_FONT_FAMILY = configure_korean_matplotlib_font()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - show all attempted encodings
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("CSV를 읽지 못했습니다.\n" + "\n".join(errors))


def parse_camera_mode(image_path: object) -> str:
    if pd.isna(image_path):
        return "unknown"
    name = Path(str(image_path)).name
    match = CAMERA_PATTERN.search(name)
    if match:
        return str(match.group(1))
    all_brackets = re.findall(r"\[([^\[\]]+)\]", name)
    if all_brackets:
        return str(all_brackets[-1])
    return "unknown"


def classify_defect_type(group_value: object) -> str:
    group = "" if pd.isna(group_value) else str(group_value)
    if TRUE_DEFECT_KEYWORD in group:
        return "진불스크래치"
    if MICRO_SCRATCH_KEYWORD in group:
        return "미세스크래치"
    return "기타"


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


class ImageViewer:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.window: tk.Toplevel | None = None
        self.meta_label: ttk.Label | None = None
        self.adjusted_image_label: ttk.Label | None = None
        self.cluster_image_label: ttk.Label | None = None
        self.cluster_stats_label: ttk.Label | None = None
        self.adjusted_photo: ImageTk.PhotoImage | None = None
        self.cluster_photo: ImageTk.PhotoImage | None = None
        self.original_array: np.ndarray | None = None
        self.raw_array: np.ndarray | None = None
        self.mask_array: np.ndarray | None = None
        self.current_path: Path | None = None
        self.current_meta = ""
        self.source_paths: dict[str, Path] = {}
        self.source_var = tk.StringVar(value="image_path")
        self.source_combo: ttk.Combobox | None = None
        self.render_after_id: str | None = None
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.DoubleVar(value=0.0)
        self.cluster_k_var = tk.IntVar(value=4)
        self.directional_cancel_var = tk.BooleanVar(value=False)
        self.directional_strength_var = tk.DoubleVar(value=0.8)
        self.directional_radius_var = tk.IntVar(value=8)
        self.cluster_mask_only_var = tk.BooleanVar(value=True)

    def _ensure_window(self) -> None:
        if self.window is not None and self.window.winfo_exists():
            return

        self.window = tk.Toplevel(self.master)
        self.window.title("선택 패치 분석")
        self.window.geometry("1580x900")
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        self.meta_label = ttk.Label(self.window, text="", anchor="w", justify="left")
        self.meta_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        control_frame = ttk.LabelFrame(self.window, text="시각화 조정", padding=(8, 6))
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(control_frame, text="Source").pack(side=tk.LEFT, padx=(0, 6))
        self.source_combo = ttk.Combobox(
            control_frame,
            textvariable=self.source_var,
            state="readonly",
            width=14,
            values=[],
        )
        self.source_combo.pack(side=tk.LEFT, padx=(0, 18))
        self.source_combo.bind("<<ComboboxSelected>>", lambda _event: self.load_selected_source())

        ttk.Label(control_frame, text="Contrast").pack(side=tk.LEFT, padx=(0, 6))
        contrast_scale = tk.Scale(
            control_frame,
            from_=0.2,
            to=3.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            length=260,
            command=lambda _value: self.schedule_render(),
        )
        contrast_scale.pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(control_frame, text="Gaussian blur").pack(side=tk.LEFT, padx=(0, 6))
        blur_scale = tk.Scale(
            control_frame,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.blur_var,
            length=220,
            command=lambda _value: self.schedule_render(),
        )
        blur_scale.pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(control_frame, text="K 그룹").pack(side=tk.LEFT, padx=(0, 6))
        k_scale = tk.Scale(
            control_frame,
            from_=2,
            to=8,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.cluster_k_var,
            length=190,
            command=lambda _value: self.schedule_render(),
        )
        k_scale.pack(side=tk.LEFT, padx=(0, 18))
        ttk.Checkbutton(
            control_frame,
            text="수직 상쇄",
            variable=self.directional_cancel_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(control_frame, text="강도").pack(side=tk.LEFT, padx=(0, 4))
        cancel_strength_scale = tk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.directional_strength_var,
            length=130,
            command=lambda _value: self.schedule_render(),
        )
        cancel_strength_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(control_frame, text="탐색").pack(side=tk.LEFT, padx=(0, 4))
        cancel_radius_scale = tk.Scale(
            control_frame,
            from_=2,
            to=24,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.directional_radius_var,
            length=130,
            command=lambda _value: self.schedule_render(),
        )
        cancel_radius_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            control_frame,
            text="Mask 영역 K-Means",
            variable=self.cluster_mask_only_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(control_frame, text="Reset", command=self.reset_controls).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(control_frame, text="K-Means는 보정/contrast/blur 적용 후 RGB 픽셀값 기준").pack(side=tk.LEFT)

        view_frame = ttk.Frame(self.window)
        view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)
        view_frame.rowconfigure(1, weight=1)

        ttk.Label(view_frame, text="Contrast 조정 이미지", anchor="center").grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Label(view_frame, text="K-Means 그룹 결과", anchor="center").grid(row=0, column=1, sticky="ew", padx=4)

        self.adjusted_image_label = ttk.Label(view_frame, anchor="center")
        self.adjusted_image_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.cluster_image_label = ttk.Label(view_frame, anchor="center")
        self.cluster_image_label.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        self.cluster_stats_label = ttk.Label(self.window, text="", anchor="w", justify="left")
        self.cluster_stats_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

    def show_error(self, message: str) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None
        self.meta_label.config(text=message)
        if self.adjusted_image_label is not None:
            self.adjusted_image_label.config(image="", text="이미지를 불러오지 못했습니다.")
        if self.cluster_image_label is not None:
            self.cluster_image_label.config(image="", text="")
        if self.cluster_stats_label is not None:
            self.cluster_stats_label.config(text="")
        self.window.deiconify()
        self.window.lift()

    def update(self, image_paths: dict[str, Path], meta_text: str) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None

        if not image_paths:
            self.show_error(f"{meta_text}\n\n분석 가능한 이미지 경로가 없습니다.")
            return

        previous_source = self.source_var.get()
        self.source_paths = image_paths
        if self.source_combo is not None:
            self.source_combo["values"] = list(image_paths.keys())
        if previous_source in image_paths:
            self.source_var.set(previous_source)
        elif "image_path" in image_paths:
            self.source_var.set("image_path")
        elif "mask_raw_path" in image_paths:
            self.source_var.set("mask_raw_path")
        else:
            self.source_var.set(next(iter(image_paths.keys())))

        self.current_meta = meta_text
        self.load_selected_source()
        self.window.deiconify()
        self.window.lift()

    def load_selected_source(self) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None

        source_name = self.source_var.get()
        image_path = self.source_paths.get(source_name)
        if image_path is None:
            self.show_error(f"{self.current_meta}\n\n선택한 source 경로를 찾지 못했습니다: {source_name}")
            return

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                self.original_array = np.asarray(img, dtype=np.uint8).copy()
            self.raw_array = self.load_rgb_array(self.source_paths.get("image_path"), self.original_array.shape[:2])
            if self.raw_array is None and source_name == "image_path":
                self.raw_array = self.original_array
            self.mask_array = self.load_mask_array(self.source_paths.get("mask_path"), self.original_array.shape[:2])
            if self.mask_array is None and source_name in {"mask_raw_path", "mask_path"}:
                self.mask_array = self.infer_nonzero_mask(self.original_array)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"{self.current_meta}\n\n이미지 로딩 실패: [{source_name}] {image_path}\n{exc}")
            return

        self.current_path = image_path
        self.meta_label.config(text=f"{self.current_meta}\nsource: {source_name}\nimage: {image_path}")
        self.render_images()

    def reset_controls(self) -> None:
        self.contrast_var.set(1.0)
        self.blur_var.set(0.0)
        self.cluster_k_var.set(4)
        self.directional_cancel_var.set(False)
        self.directional_strength_var.set(0.8)
        self.directional_radius_var.set(8)
        self.cluster_mask_only_var.set(True)
        self.render_images()

    def schedule_render(self) -> None:
        if self.window is None or not self.window.winfo_exists():
            return
        if self.render_after_id is not None:
            self.window.after_cancel(self.render_after_id)
        self.render_after_id = self.window.after(120, self.render_images)

    def render_images(self) -> None:
        self.render_after_id = None
        if self.original_array is None:
            return
        assert self.adjusted_image_label is not None and self.cluster_image_label is not None

        contrast = float(self.contrast_var.get())
        blur_radius = float(self.blur_var.get())
        k = int(self.cluster_k_var.get())
        base, correction_text = self.make_analysis_base()
        adjusted = self.apply_blur(self.apply_contrast(base, contrast), blur_radius)
        cluster_mask = self.mask_array if self.cluster_mask_only_var.get() else None
        clustered, stats_text = self.kmeans_cluster_image(adjusted, k, mask=cluster_mask)

        self.adjusted_photo = ImageTk.PhotoImage(self.to_display_image(adjusted))
        self.cluster_photo = ImageTk.PhotoImage(self.to_display_image(clustered))
        self.adjusted_image_label.config(image=self.adjusted_photo, text="")
        self.cluster_image_label.config(image=self.cluster_photo, text="")
        if self.cluster_stats_label is not None:
            self.cluster_stats_label.config(
                text=f"contrast={contrast:.2f}, blur={blur_radius:.1f}, K={k}, mask_only={self.cluster_mask_only_var.get()} | {correction_text} | {stats_text}"
            )

    def make_analysis_base(self) -> tuple[np.ndarray, str]:
        assert self.original_array is not None
        source_name = self.source_var.get()
        if not self.directional_cancel_var.get():
            return self.original_array, "directional_cancel=off"
        if self.raw_array is None or self.mask_array is None:
            return self.original_array, "directional_cancel=unavailable(raw/mask missing)"

        strength = float(self.directional_strength_var.get())
        radius = int(self.directional_radius_var.get())
        mask_only_output = source_name in {"mask_raw_path", "mask_path"}
        corrected, stats = self.directional_perpendicular_cancel(
            raw_array=self.raw_array,
            mask=self.mask_array,
            strength=strength,
            search_radius=radius,
            mask_only_output=mask_only_output,
        )
        return corrected, f"directional_cancel=on, strength={strength:.2f}, radius={radius}, {stats}"

    @staticmethod
    def apply_contrast(array: np.ndarray, contrast: float) -> np.ndarray:
        arr = array.astype(np.float32)
        adjusted = (arr - 127.5) * contrast + 127.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_blur(array: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return array
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.uint8)

    @staticmethod
    def load_rgb_array(path: Path | None, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        with Image.open(path) as img:
            img = img.convert("RGB")
            if expected_shape is not None and img.size != (expected_shape[1], expected_shape[0]):
                img = img.resize((expected_shape[1], expected_shape[0]), Image.Resampling.BILINEAR)
            return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()

    @staticmethod
    def load_mask_array(path: Path | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        with Image.open(path) as img:
            mask_img = img.convert("L")
            if mask_img.size != (expected_shape[1], expected_shape[0]):
                mask_img = mask_img.resize((expected_shape[1], expected_shape[0]), Image.Resampling.NEAREST)
            return np.asarray(mask_img, dtype=np.uint8) > 0

    @staticmethod
    def infer_nonzero_mask(array: np.ndarray) -> np.ndarray:
        return np.any(array.astype(np.uint8) > 0, axis=2)

    @staticmethod
    def connected_components_bool(mask: np.ndarray, min_area: int = 8) -> list[np.ndarray]:
        mask_bool = mask.astype(bool)
        h, w = mask_bool.shape
        visited = np.zeros_like(mask_bool, dtype=bool)
        components: list[np.ndarray] = []
        for y0, x0 in zip(*np.where(mask_bool & ~visited)):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(coords) >= min_area:
                components.append(np.asarray(coords, dtype=np.int32))
        return components

    @staticmethod
    def component_perpendicular(coords_yx: np.ndarray) -> tuple[float, float]:
        coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
        if len(coords_xy) < 2:
            return 0.0, 1.0
        centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, int(np.argmax(eigvals))]
        perp_x = -float(major_vec[1])
        perp_y = float(major_vec[0])
        norm = math.sqrt(perp_x * perp_x + perp_y * perp_y)
        if norm < 1e-6:
            return 0.0, 1.0
        return perp_y / norm, perp_x / norm

    @staticmethod
    def sample_perpendicular_background(
        raw_array: np.ndarray,
        mask: np.ndarray,
        y: int,
        x: int,
        perp_y: float,
        perp_x: float,
        search_radius: int,
    ) -> np.ndarray | None:
        h, w = mask.shape
        samples = []
        for sign in (-1, 1):
            for distance in range(1, search_radius + 1):
                ny = int(round(y + sign * perp_y * distance))
                nx = int(round(x + sign * perp_x * distance))
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if not mask[ny, nx]:
                    samples.append(raw_array[ny, nx].astype(np.float32))
                    break
        if not samples:
            return None
        return np.mean(np.vstack(samples), axis=0)

    @classmethod
    def directional_perpendicular_cancel(
        cls,
        raw_array: np.ndarray,
        mask: np.ndarray,
        strength: float,
        search_radius: int,
        mask_only_output: bool,
    ) -> tuple[np.ndarray, str]:
        if raw_array.shape[:2] != mask.shape:
            raise ValueError("raw_array and mask shape mismatch")
        strength = float(np.clip(strength, 0.0, 1.0))
        search_radius = int(np.clip(search_radius, 1, 64))

        output = np.zeros_like(raw_array) if mask_only_output else raw_array.copy()
        components = cls.connected_components_bool(mask, min_area=8)
        corrected_pixels = 0
        for coords in components:
            perp_y, perp_x = cls.component_perpendicular(coords)
            for y, x in coords:
                replacement = cls.sample_perpendicular_background(raw_array, mask, int(y), int(x), perp_y, perp_x, search_radius)
                if replacement is None:
                    output[y, x] = raw_array[y, x]
                    continue
                current = raw_array[y, x].astype(np.float32)
                output[y, x] = np.clip((1.0 - strength) * current + strength * replacement, 0, 255).astype(np.uint8)
                corrected_pixels += 1
        return output, f"components={len(components)}, corrected_px={corrected_pixels}"

    @staticmethod
    def to_display_image(array: np.ndarray) -> Image.Image:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        image.thumbnail((550, 650), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def kmeans_cluster_image(array: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        k = int(np.clip(k, 2, 8))
        h, w, _ = array.shape
        if mask is not None and mask.shape == (h, w) and mask.any():
            valid_mask = mask.astype(bool)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        pixels = array[valid_mask].reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(17)
        sample_size = min(25000, len(pixels))
        if sample_size <= 0:
            return np.zeros_like(array), "no valid pixels"
        sample_idx = rng.choice(len(pixels), size=sample_size, replace=False)
        sample = pixels[sample_idx]

        luma = sample @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        quantiles = np.linspace(0.05, 0.95, k)
        centers = []
        for q in quantiles:
            target = np.quantile(luma, q)
            centers.append(sample[int(np.argmin(np.abs(luma - target)))])
        centers = np.asarray(centers, dtype=np.float32)

        for _ in range(14):
            distances = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_id in range(k):
                cluster_pixels = sample[labels == cluster_id]
                if len(cluster_pixels):
                    new_centers[cluster_id] = cluster_pixels.mean(axis=0)
                else:
                    new_centers[cluster_id] = sample[rng.integers(0, len(sample))]
            if np.allclose(new_centers, centers, atol=0.5):
                centers = new_centers
                break
            centers = new_centers

        all_labels = np.empty(len(pixels), dtype=np.int16)
        chunk = 100000
        for start in range(0, len(pixels), chunk):
            stop = min(start + chunk, len(pixels))
            distances = ((pixels[start:stop, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            all_labels[start:stop] = distances.argmin(axis=1)

        center_luma = centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        order = np.argsort(center_luma)
        remap = np.zeros(k, dtype=np.int16)
        for new_id, old_id in enumerate(order):
            remap[old_id] = new_id
        ordered_labels = remap[all_labels]
        ordered_centers = centers[order]

        palette = np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
            ],
            dtype=np.uint8,
        )
        clustered = np.zeros((h, w, 3), dtype=np.uint8)
        clustered[valid_mask] = palette[ordered_labels]
        counts = np.bincount(ordered_labels, minlength=k)
        total = max(int(counts.sum()), 1)
        stats = []
        for cluster_id in range(k):
            mean_rgb = ordered_centers[cluster_id]
            rate = counts[cluster_id] / total * 100
            stats.append(
                f"C{cluster_id}: {rate:.1f}%, RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})"
            )
        return clustered, " | ".join(stats)


class GroupCameraFeatureExplorer:
    def __init__(self, root: tk.Tk, csv_path: Path | None = None, image_root: Path | None = None) -> None:
        self.root = root
        self._configure_tk_font()
        self.root.title("Group/Camera Feature Explorer")
        self.root.geometry("1440x900")

        self.csv_path: Path | None = None
        self.csv_dir: Path | None = None
        self.image_root: Path | None = image_root
        self.image_search_cache: dict[str, Path | None] = {}
        self.df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.numeric_features: list[str] = []
        self.custom_terms: list[dict[str, object]] = []
        self.custom_metric_counter = 1
        self.artist_rows: dict[object, list[int]] = {}
        self.image_viewer = ImageViewer(root)

        self.csv_path_var = tk.StringVar(value=str(csv_path) if csv_path else "")
        self.image_root_var = tk.StringVar(value=str(image_root) if image_root else "")
        self.feature_var = tk.StringVar()
        self.group_filter_var = tk.StringVar(value="전체")
        self.camera_filter_var = tk.StringVar(value="전체")
        self.sort_var = tk.StringVar(value="group_camera")
        self.formula_feature_var = tk.StringVar()
        self.formula_weight_var = tk.StringVar(value="1.0")
        self.formula_norm_var = tk.StringVar(value="zscore")
        self.formula_name_var = tk.StringVar(value="custom_metric")
        self.formula_text_var = tk.StringVar(value="항을 추가하세요.")
        self.threshold_var = tk.StringVar()
        self.threshold_direction_var = tk.StringVar(value="상단 제거 (>=)")
        self.status_var = tk.StringVar(value="CSV를 로드하세요.")

        self._build_layout()
        if csv_path is not None:
            self.load_csv(csv_path)

    def _configure_tk_font(self) -> None:
        if not KOREAN_FONT_FAMILY:
            return
        for font_name in [
            "TkDefaultFont",
            "TkTextFont",
            "TkFixedFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ]:
            try:
                current_font = tkfont.nametofont(font_name)
                current_font.configure(family=KOREAN_FONT_FAMILY)
            except tk.TclError:
                continue

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="CSV 열기", command=self.choose_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.csv_path_var, width=70).pack(side=tk.LEFT, padx=(0, 12), fill=tk.X, expand=True)

        ttk.Button(top, text="이미지 루트", command=self.choose_image_root).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.image_root_var, width=38).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(top, text="다시 그리기", command=self.refresh_plot).pack(side=tk.LEFT)

        controls = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Feature").pack(side=tk.LEFT)
        self.feature_combo = ttk.Combobox(controls, textvariable=self.feature_var, state="readonly", width=36)
        self.feature_combo.pack(side=tk.LEFT, padx=(6, 18))
        self.feature_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())

        ttk.Label(controls, text="Group").pack(side=tk.LEFT)
        self.group_combo = ttk.Combobox(controls, textvariable=self.group_filter_var, state="readonly", width=28)
        self.group_combo.pack(side=tk.LEFT, padx=(6, 18))
        self.group_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())

        ttk.Label(controls, text="Camera").pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(controls, textvariable=self.camera_filter_var, state="readonly", width=12)
        self.camera_combo.pack(side=tk.LEFT, padx=(6, 18))
        self.camera_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())

        ttk.Label(controls, text="정렬").pack(side=tk.LEFT)
        self.sort_combo = ttk.Combobox(
            controls,
            textvariable=self.sort_var,
            state="readonly",
            width=16,
            values=["group_camera", "mean_desc", "mean_asc", "count_desc"],
        )
        self.sort_combo.pack(side=tk.LEFT, padx=(6, 18))
        self.sort_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())

        formula_frame = ttk.LabelFrame(self.root, text="사용자 정의 지표", padding=(8, 6))
        formula_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        formula_top = ttk.Frame(formula_frame)
        formula_top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(formula_top, text="항 Feature").pack(side=tk.LEFT)
        self.formula_feature_combo = ttk.Combobox(formula_top, textvariable=self.formula_feature_var, state="readonly", width=34)
        self.formula_feature_combo.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(formula_top, text="가중치").pack(side=tk.LEFT)
        ttk.Entry(formula_top, textvariable=self.formula_weight_var, width=10).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(formula_top, text="정규화").pack(side=tk.LEFT)
        self.formula_norm_combo = ttk.Combobox(
            formula_top,
            textvariable=self.formula_norm_var,
            state="readonly",
            width=12,
            values=["zscore", "robust", "minmax", "none"],
        )
        self.formula_norm_combo.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(formula_top, text="항 추가", command=self.add_formula_term).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(formula_top, text="선택 삭제", command=self.delete_selected_formula_terms).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(formula_top, text="전체 삭제", command=self.clear_formula_terms).pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(formula_top, text="지표명").pack(side=tk.LEFT)
        ttk.Entry(formula_top, textvariable=self.formula_name_var, width=20).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(formula_top, text="지표 적용", command=self.apply_custom_metric).pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(formula_top, text="Threshold").pack(side=tk.LEFT)
        ttk.Entry(formula_top, textvariable=self.threshold_var, width=12).pack(side=tk.LEFT, padx=(6, 6))
        self.threshold_direction_combo = ttk.Combobox(
            formula_top,
            textvariable=self.threshold_direction_var,
            state="readonly",
            width=16,
            values=["상단 제거 (>=)", "하단 제거 (<=)"],
        )
        self.threshold_direction_combo.pack(side=tk.LEFT, padx=(0, 6))
        self.threshold_direction_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())
        ttk.Button(formula_top, text="Threshold 적용", command=self.refresh_plot).pack(side=tk.LEFT)

        formula_bottom = ttk.Frame(formula_frame)
        formula_bottom.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        self.term_listbox = tk.Listbox(formula_bottom, height=3, width=72, exportselection=False)
        self.term_listbox.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=(0, 8))
        ttk.Label(formula_bottom, textvariable=self.formula_text_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        plot_frame = ttk.Frame(main)
        table_frame = ttk.Frame(main)
        main.add(plot_frame, weight=4)
        main.add(table_frame, weight=2)

        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(table_frame, text="Group x Camera 평균/분포 요약").pack(side=tk.TOP, anchor="w")
        columns = [
            "group",
            "camera_mode",
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "true_defect_count",
            "micro_count",
            "other_count",
        ]
        self.summary_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=22)
        for col in columns:
            self.summary_tree.heading(col, text=col)
            width = 180 if col == "group" else 105
            self.summary_tree.column(col, width=width, minwidth=80, anchor="center")

        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.summary_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(table_frame, text="Threshold 상단/하단 요약").pack(side=tk.TOP, anchor="w", pady=(8, 0))
        threshold_columns = [
            "scope",
            "side",
            "camera_mode",
            "defect_type",
            "count",
            "rate_total",
            "rate_within_type",
            "mean",
        ]
        self.threshold_tree = ttk.Treeview(table_frame, columns=threshold_columns, show="headings", height=10)
        for col in threshold_columns:
            self.threshold_tree.heading(col, text=col)
            width = 120
            if col in {"scope", "defect_type"}:
                width = 135
            if col == "camera_mode":
                width = 90
            self.threshold_tree.column(col, width=width, minwidth=70, anchor="center")
        self.threshold_tree.pack(side=tk.TOP, fill=tk.X)

        self.status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 2))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def choose_csv(self) -> None:
        initial = str(self.csv_path.parent) if self.csv_path else str(Path.cwd())
        selected = filedialog.askopenfilename(
            title="Feature CSV 선택",
            initialdir=initial,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.load_csv(Path(selected))

    def choose_image_root(self) -> None:
        initial = self.image_root_var.get() or str(self.csv_dir or Path.cwd())
        selected = filedialog.askdirectory(title="이미지 루트 폴더 선택", initialdir=initial)
        if selected:
            self.image_root = Path(selected)
            self.image_root_var.set(str(self.image_root))
            self.image_search_cache.clear()
            self.status_var.set(f"이미지 루트 설정: {self.image_root}")

    def add_formula_term(self) -> None:
        if self.df.empty:
            messagebox.showwarning("CSV 필요", "먼저 CSV를 로드하세요.")
            return

        feature = self.formula_feature_var.get()
        if not feature or feature not in self.numeric_features:
            messagebox.showwarning("Feature 선택", "수치형 feature를 선택하세요.")
            return

        try:
            weight = float(self.formula_weight_var.get())
        except ValueError:
            messagebox.showwarning("가중치 오류", "가중치는 숫자로 입력하세요. 예: 1, -0.5, 2.3")
            return

        if not np.isfinite(weight):
            messagebox.showwarning("가중치 오류", "가중치는 유한한 숫자여야 합니다.")
            return

        norm = self.formula_norm_var.get() or "zscore"
        self.custom_terms.append({"feature": feature, "weight": weight, "norm": norm})
        self.update_formula_display()

    def delete_selected_formula_terms(self) -> None:
        selected = list(self.term_listbox.curselection())
        if not selected:
            return
        for idx in sorted(selected, reverse=True):
            if 0 <= idx < len(self.custom_terms):
                self.custom_terms.pop(idx)
        self.update_formula_display()

    def clear_formula_terms(self) -> None:
        self.custom_terms.clear()
        self.update_formula_display()

    def update_formula_display(self) -> None:
        if not hasattr(self, "term_listbox"):
            return
        self.term_listbox.delete(0, tk.END)
        if not self.custom_terms:
            self.formula_text_var.set("항을 추가하세요. 예: bbox_minor * 1 + bbox_major * 0.3 - contrast * 0.8")
            return

        parts = []
        for term in self.custom_terms:
            feature = str(term["feature"])
            weight = float(term["weight"])
            norm = str(term["norm"])
            sign = "+" if weight >= 0 else "-"
            abs_weight = abs(weight)
            text = f"{sign} {abs_weight:g} * {norm}({feature})"
            parts.append(text)
            self.term_listbox.insert(tk.END, text.lstrip("+ "))
        self.formula_text_var.set("custom = " + " ".join(parts).lstrip("+ "))

    @staticmethod
    def normalize_series(values: pd.Series, norm: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        if norm == "none":
            return numeric
        if norm == "minmax":
            vmin = numeric.min(skipna=True)
            vmax = numeric.max(skipna=True)
            denom = vmax - vmin
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                return numeric * 0.0
            return (numeric - vmin) / denom
        if norm == "robust":
            median = numeric.median(skipna=True)
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or abs(iqr) < 1e-12:
                return numeric * 0.0
            return (numeric - median) / iqr

        mean = numeric.mean(skipna=True)
        std = numeric.std(skipna=True)
        if not np.isfinite(std) or abs(std) < 1e-12:
            return numeric * 0.0
        return (numeric - mean) / std

    def apply_custom_metric(self) -> None:
        if self.df.empty:
            messagebox.showwarning("CSV 필요", "먼저 CSV를 로드하세요.")
            return
        if not self.custom_terms:
            messagebox.showwarning("항 없음", "지표를 만들 항을 하나 이상 추가하세요.")
            return

        metric_name = self.formula_name_var.get().strip() or f"custom_metric_{self.custom_metric_counter}"
        metric_name = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", metric_name).strip("_")
        if not metric_name:
            metric_name = f"custom_metric_{self.custom_metric_counter}"
        if metric_name in ID_LIKE_COLUMNS:
            metric_name = f"{metric_name}_metric"

        values = pd.Series(0.0, index=self.df.index, dtype=float)
        missing_features = []
        for term in self.custom_terms:
            feature = str(term["feature"])
            weight = float(term["weight"])
            norm = str(term["norm"])
            if feature not in self.df.columns:
                missing_features.append(feature)
                continue
            feature_values = self.normalize_series(self.df[feature], norm)
            values = values.add(feature_values * weight, fill_value=0.0)

        if missing_features:
            messagebox.showwarning("누락 feature", f"없는 feature가 있어 제외했습니다: {missing_features}")

        self.df[metric_name] = values.replace([np.inf, -np.inf], np.nan)
        if metric_name not in self.numeric_features:
            self.numeric_features.append(metric_name)
        self.feature_combo["values"] = self.numeric_features
        self.formula_feature_combo["values"] = self.numeric_features
        self.feature_var.set(metric_name)
        self.custom_metric_counter += 1
        self.formula_name_var.set(f"custom_metric_{self.custom_metric_counter}")
        self.status_var.set(f"사용자 정의 지표 생성: {metric_name} = {self.formula_text_var.get()}")
        self.refresh_plot()

    def load_csv(self, path: Path) -> None:
        try:
            df = read_csv_flexible(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("CSV 로딩 실패", str(exc))
            return

        required = {"image_path", "group"}
        missing = sorted(required - set(df.columns))
        if missing:
            messagebox.showerror("CSV 컬럼 부족", f"필수 컬럼이 없습니다: {missing}")
            return

        self.csv_path = path
        self.csv_dir = path.parent
        self.csv_path_var.set(str(path))
        self.df = self._prepare_dataframe(df)
        self.numeric_features = self._numeric_feature_columns(self.df)
        self.custom_terms.clear()
        self.custom_metric_counter = 1

        if not self.numeric_features:
            messagebox.showerror("수치형 feature 없음", "그래프로 표시할 numeric feature가 없습니다.")
            return

        self.feature_combo["values"] = self.numeric_features
        self.formula_feature_combo["values"] = self.numeric_features
        default_feature = self._default_feature(self.numeric_features)
        self.feature_var.set(default_feature)
        self.formula_feature_var.set(default_feature)
        self.formula_norm_var.set("zscore")
        self.formula_name_var.set("custom_metric")
        self.threshold_var.set("")
        self.update_formula_display()

        groups = ["전체"] + sorted(self.df["group"].astype(str).dropna().unique().tolist())
        cameras = ["전체"] + sorted(self.df["camera_mode"].astype(str).dropna().unique().tolist(), key=self._camera_sort_key)
        self.group_combo["values"] = groups
        self.camera_combo["values"] = cameras
        self.group_filter_var.set("전체")
        self.camera_filter_var.set("전체")

        self.status_var.set(f"로드 완료: rows={len(self.df)}, numeric_features={len(self.numeric_features)}")
        self.refresh_plot()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["row_id"] = np.arange(len(out))
        out["group"] = out["group"].astype(str)
        out["camera_mode"] = out["image_path"].map(parse_camera_mode)
        out["defect_type"] = out["group"].map(classify_defect_type)

        for col in out.columns:
            if col in ID_LIKE_COLUMNS:
                continue
            out[col] = pd.to_numeric(out[col], errors="ignore")

        area_col = "area" if "area" in out.columns else "area_px" if "area_px" in out.columns else None
        width_col = "bbox_width" if "bbox_width" in out.columns else "bbox_w" if "bbox_w" in out.columns else None
        height_col = "bbox_height" if "bbox_height" in out.columns else "bbox_h" if "bbox_h" in out.columns else None

        if width_col and height_col:
            width = pd.to_numeric(out[width_col], errors="coerce").clip(lower=0)
            height = pd.to_numeric(out[height_col], errors="coerce").clip(lower=0)
            out["bbox_major"] = np.maximum(width, height)
            out["bbox_minor"] = np.minimum(width, height)
            out["bbox_aspect_engineered"] = out["bbox_major"] / (out["bbox_minor"] + 1e-6)
            out["bbox_area_engineered"] = width * height
            if area_col:
                area = pd.to_numeric(out[area_col], errors="coerce").clip(lower=0)
                out["bbox_fill_ratio_engineered"] = area / (out["bbox_area_engineered"] + 1e-6)
                out["log_area"] = np.log1p(area)

        rgb_std_cols = [col for col in ["rgb_std_b", "rgb_std_g", "rgb_std_r"] if col in out.columns]
        if rgb_std_cols:
            rgb_std = out[rgb_std_cols].apply(pd.to_numeric, errors="coerce")
            out["rgb_std_mean"] = rgb_std.mean(axis=1)
            out["rgb_std_max"] = rgb_std.max(axis=1)

        return out

    @staticmethod
    def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        for col in df.columns:
            if col in ID_LIKE_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 0 and df[col].nunique(dropna=True) > 1:
                cols.append(col)
        return cols

    @staticmethod
    def _default_feature(features: list[str]) -> str:
        for candidate in [
            "area",
            "bbox_minor",
            "bbox_aspect_engineered",
            "perimeter_skeleton_diff_euclidean",
            "rgb_contrast_z",
        ]:
            if candidate in features:
                return candidate
        return features[0]

    @staticmethod
    def _camera_sort_key(value: object) -> tuple[int, object]:
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    def filtered(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        part = self.df.copy()
        group_value = self.group_filter_var.get()
        camera_value = self.camera_filter_var.get()
        if group_value and group_value != "전체":
            part = part[part["group"].astype(str) == group_value]
        if camera_value and camera_value != "전체":
            part = part[part["camera_mode"].astype(str) == camera_value]
        return part

    def parse_threshold(self) -> float | None:
        raw = self.threshold_var.get().strip()
        if not raw:
            return None
        try:
            threshold = float(raw)
        except ValueError:
            return None
        if not np.isfinite(threshold):
            return None
        return threshold

    def threshold_filter_mask(self, values: pd.Series, threshold: float) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if self.threshold_direction_var.get().startswith("하단"):
            return numeric <= threshold
        return numeric >= threshold

    def clear_threshold_table(self) -> None:
        if not hasattr(self, "threshold_tree"):
            return
        for item in self.threshold_tree.get_children():
            self.threshold_tree.delete(item)

    def update_threshold_table(self, plot_df: pd.DataFrame, feature: str, threshold: float | None) -> None:
        self.clear_threshold_table()
        if threshold is None or plot_df.empty:
            return

        work = plot_df.copy()
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work = work.dropna(subset=[feature])
        if work.empty:
            return

        work["side"] = np.where(work[feature] >= threshold, "상단(>=)", "하단(<)")
        total = len(work)
        type_total = work.groupby("defect_type").size().to_dict()

        rows = []
        for side, part in work.groupby("side", sort=False):
            rows.append(
                {
                    "scope": "전체",
                    "side": side,
                    "camera_mode": "전체",
                    "defect_type": "전체",
                    "count": len(part),
                    "rate_total": len(part) / max(total, 1),
                    "rate_within_type": len(part) / max(total, 1),
                    "mean": part[feature].mean(),
                }
            )

        for (side, defect_type), part in work.groupby(["side", "defect_type"], sort=False):
            rows.append(
                {
                    "scope": "라벨별",
                    "side": side,
                    "camera_mode": "전체",
                    "defect_type": defect_type,
                    "count": len(part),
                    "rate_total": len(part) / max(total, 1),
                    "rate_within_type": len(part) / max(type_total.get(defect_type, 0), 1),
                    "mean": part[feature].mean(),
                }
            )

        camera_type_total = work.groupby(["camera_mode", "defect_type"]).size().to_dict()
        for (side, camera_mode, defect_type), part in work.groupby(["side", "camera_mode", "defect_type"], sort=False):
            denom = camera_type_total.get((camera_mode, defect_type), 0)
            rows.append(
                {
                    "scope": "카메라x라벨",
                    "side": side,
                    "camera_mode": camera_mode,
                    "defect_type": defect_type,
                    "count": len(part),
                    "rate_total": len(part) / max(total, 1),
                    "rate_within_type": len(part) / max(denom, 1),
                    "mean": part[feature].mean(),
                }
            )

        sort_key = {"전체": 0, "라벨별": 1, "카메라x라벨": 2}
        rows = sorted(rows, key=lambda row: (sort_key.get(row["scope"], 99), row["side"], str(row["camera_mode"]), str(row["defect_type"])))
        for row in rows:
            values = [
                row["scope"],
                row["side"],
                row["camera_mode"],
                row["defect_type"],
                int(row["count"]),
                self.format_percent(row["rate_total"]),
                self.format_percent(row["rate_within_type"]),
                self.format_number(row["mean"]),
            ]
            self.threshold_tree.insert("", tk.END, values=values)

    def refresh_plot(self) -> None:
        if self.df.empty:
            return
        feature = self.feature_var.get()
        if not feature or feature not in self.df.columns:
            return

        self.filtered_df = self.filtered().copy()
        plot_df = self.filtered_df[["row_id", "group", "camera_mode", "defect_type", "image_path", feature]].copy()
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.dropna(subset=[feature])
        threshold = self.parse_threshold()

        self.ax.clear()
        self.artist_rows.clear()
        self.update_summary_table(plot_df, feature)
        self.update_threshold_table(plot_df, feature, threshold)

        if plot_df.empty:
            self.ax.set_title("표시할 데이터가 없습니다.")
            self.canvas.draw_idle()
            self.status_var.set("필터 결과가 비어 있습니다.")
            return

        summary = self.summary_frame(plot_df, feature)
        categories = self.sorted_categories(summary)
        cat_to_x = {cat: idx for idx, cat in enumerate(categories)}
        plot_df["category"] = list(zip(plot_df["group"].astype(str), plot_df["camera_mode"].astype(str)))
        plot_df["x_base"] = plot_df["category"].map(cat_to_x)

        rng = np.random.default_rng(7)
        plot_df["x"] = plot_df["x_base"].astype(float) + rng.uniform(-0.22, 0.22, size=len(plot_df))

        for defect_type, part in plot_df.groupby("defect_type", sort=False):
            color = DEFECT_COLORS.get(defect_type, "#666666")
            artist = self.ax.scatter(
                part["x"],
                part[feature],
                s=34,
                alpha=0.72,
                label=f"{defect_type} ({len(part)})",
                color=color,
                edgecolors="white",
                linewidths=0.45,
                picker=True,
                pickradius=6,
            )
            self.artist_rows[artist] = part["row_id"].astype(int).tolist()

        for _, row in summary.iterrows():
            cat = (str(row["group"]), str(row["camera_mode"]))
            x = cat_to_x.get(cat)
            if x is None:
                continue
            self.ax.scatter(x, row["mean"], marker="D", s=58, color="black", zorder=5)
            self.ax.hlines(row["mean"], x - 0.32, x + 0.32, colors="black", linewidth=2.0, zorder=4)

        threshold_status = ""
        if threshold is not None:
            threshold_mask = self.threshold_filter_mask(plot_df[feature], threshold)
            filtered_by_threshold = plot_df[threshold_mask]
            upper_count = int((plot_df[feature] >= threshold).sum())
            lower_count = int((plot_df[feature] < threshold).sum())
            direction_text = self.threshold_direction_var.get()
            threshold_status = (
                f" | threshold={threshold:g}, 상단={upper_count}, 하단={lower_count}, "
                f"{direction_text} 필터={len(filtered_by_threshold)}"
            )
            self.ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.4, label=f"threshold={threshold:g}")
            if not filtered_by_threshold.empty:
                self.ax.scatter(
                    filtered_by_threshold["x"],
                    filtered_by_threshold[feature],
                    s=92,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.3,
                    label=f"threshold filter ({len(filtered_by_threshold)})",
                    zorder=6,
                )
        elif self.threshold_var.get().strip():
            threshold_status = " | threshold 입력값이 숫자가 아닙니다."

        labels = [f"{group}\ncam={camera}" for group, camera in categories]
        self.ax.set_xticks(range(len(categories)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        self.ax.set_ylabel(feature)
        self.ax.set_xlabel("group x camera_mode")
        self.ax.set_title(f"{feature}: group/camera 분포와 평균")
        self.ax.legend(loc="best", fontsize=8)
        self.ax.grid(True, alpha=0.28)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(
            f"표시 rows={len(plot_df)} | groups={plot_df['group'].nunique()} | cameras={plot_df['camera_mode'].nunique()} | 점 클릭 시 이미지 창 갱신{threshold_status}"
        )

    def summary_frame(self, plot_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        grouped = plot_df.groupby(["group", "camera_mode"], dropna=False)
        summary = grouped[feature].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
        type_counts = (
            plot_df.pivot_table(
                index=["group", "camera_mode"],
                columns="defect_type",
                values="row_id",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        summary = summary.merge(type_counts, on=["group", "camera_mode"], how="left")
        for col in ["진불스크래치", "미세스크래치", "기타"]:
            if col not in summary.columns:
                summary[col] = 0
        return summary

    def sorted_categories(self, summary: pd.DataFrame) -> list[tuple[str, str]]:
        sort_mode = self.sort_var.get()
        work = summary.copy()
        if sort_mode == "mean_desc":
            work = work.sort_values("mean", ascending=False)
        elif sort_mode == "mean_asc":
            work = work.sort_values("mean", ascending=True)
        elif sort_mode == "count_desc":
            work = work.sort_values("count", ascending=False)
        else:
            work = work.sort_values(["group", "camera_mode"], key=lambda col: col.astype(str))
        return [(str(row["group"]), str(row["camera_mode"])) for _, row in work.iterrows()]

    def update_summary_table(self, plot_df: pd.DataFrame, feature: str) -> None:
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        if plot_df.empty:
            return

        summary = self.summary_frame(plot_df, feature)
        sort_mode = self.sort_var.get()
        if sort_mode == "mean_desc":
            summary = summary.sort_values("mean", ascending=False)
        elif sort_mode == "mean_asc":
            summary = summary.sort_values("mean", ascending=True)
        elif sort_mode == "count_desc":
            summary = summary.sort_values("count", ascending=False)
        else:
            summary = summary.sort_values(["group", "camera_mode"], key=lambda col: col.astype(str))

        for _, row in summary.iterrows():
            values = [
                row["group"],
                row["camera_mode"],
                int(row["count"]),
                self.format_number(row["mean"]),
                self.format_number(row["std"]),
                self.format_number(row["median"]),
                self.format_number(row["min"]),
                self.format_number(row["max"]),
                int(row.get("진불스크래치", 0)),
                int(row.get("미세스크래치", 0)),
                int(row.get("기타", 0)),
            ]
            self.summary_tree.insert("", tk.END, values=values)

    @staticmethod
    def format_number(value: object) -> str:
        number = safe_float(value)
        if math.isnan(number):
            return ""
        if abs(number) >= 100:
            return f"{number:.2f}"
        if abs(number) >= 1:
            return f"{number:.4f}"
        return f"{number:.6f}"

    @staticmethod
    def format_percent(value: object) -> str:
        number = safe_float(value)
        if math.isnan(number):
            return ""
        return f"{number * 100:.2f}%"

    def on_pick(self, event) -> None:  # noqa: ANN001 - matplotlib event
        artist = event.artist
        if artist not in self.artist_rows or not len(event.ind):
            return
        local_idx = int(event.ind[0])
        row_id = self.artist_rows[artist][local_idx]
        row = self.df.loc[self.df["row_id"] == row_id].iloc[0]
        self.show_row_image(row)

    def show_row_image(self, row: pd.Series) -> None:
        feature = self.feature_var.get()
        value = row.get(feature, "")
        meta = (
            f"row_id={row.get('row_id')} | component_id={row.get('component_id', '')} | "
            f"group={row.get('group')} | camera={row.get('camera_mode')} | "
            f"defect_type={row.get('defect_type')} | {feature}={self.format_number(value)}"
        )

        image_paths: dict[str, Path] = {}
        missing_sources: list[str] = []
        for source_name in ["image_path", "mask_raw_path", "mask_path"]:
            if source_name not in row.index:
                continue
            resolved = self.resolve_image_path(row.get(source_name))
            if resolved is not None:
                image_paths[source_name] = resolved
            elif not pd.isna(row.get(source_name)) and str(row.get(source_name)).strip():
                missing_sources.append(f"{source_name}={row.get(source_name)}")

        if not image_paths:
            missing_text = "\n".join(missing_sources) if missing_sources else "image_path/mask_raw_path/mask_path 값이 없습니다."
            self.image_viewer.show_error(f"{meta}\n\n이미지 파일을 찾지 못했습니다:\n{missing_text}")
            return
        self.image_viewer.update(image_paths, meta)

    def resolve_image_path(self, value: object) -> Path | None:
        if pd.isna(value):
            return None
        raw = str(value).strip().strip('"')
        if not raw:
            return None

        path = Path(raw)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if self.csv_dir is not None:
                candidates.append(self.csv_dir / path)
            candidates.append(Path.cwd() / path)
            if self.image_root is not None:
                candidates.append(self.image_root / path)
                candidates.append(self.image_root / path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        basename = path.name
        if basename in self.image_search_cache:
            return self.image_search_cache[basename]

        for root in [self.image_root, self.csv_dir]:
            if root is None or not root.exists():
                continue
            try:
                found = next(root.rglob(basename), None)
            except Exception:  # noqa: BLE001
                found = None
            if found is not None and found.exists():
                self.image_search_cache[basename] = found
                return found

        self.image_search_cache[basename] = None
        return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group/camera feature UI for scratch component CSV files.")
    parser.add_argument("--csv", type=Path, default=None, help="Feature CSV path")
    parser.add_argument("--image-root", type=Path, default=None, help="Root directory for image_path lookup")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = tk.Tk()
    GroupCameraFeatureExplorer(root, csv_path=args.csv, image_root=args.image_root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
