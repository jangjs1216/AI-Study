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
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover - optional acceleration
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError("Pillow가 필요합니다. `python -m pip install pillow` 후 다시 실행하세요.") from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
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

MAIN_FILTER_CACHE_VERSION = "main_filter_v3"
OPENCV_JBF_INSTALL_HINT = "OpenCV Joint Bilateral Filter는 opencv-contrib-python의 cv2.ximgproc가 필요합니다."

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
    def __init__(
        self,
        master: tk.Tk,
        filter_pipeline_provider: Callable[[], dict[str, object]] | None = None,
        pipeline_preview_renderer: Callable[
            [np.ndarray, np.ndarray | None, dict[str, object]],
            tuple[np.ndarray, np.ndarray, np.ndarray, str],
        ]
        | None = None,
    ) -> None:
        self.master = master
        self.filter_pipeline_provider = filter_pipeline_provider
        self.pipeline_preview_renderer = pipeline_preview_renderer
        self.window: tk.Toplevel | None = None
        self.meta_label: ttk.Label | None = None
        self.adjusted_image_label: ttk.Label | None = None
        self.cluster_image_label: ttk.Label | None = None
        self.refined_image_label: ttk.Label | None = None
        self.cluster_stats_label: ttk.Label | None = None
        self.adjusted_photo: ImageTk.PhotoImage | None = None
        self.cluster_photo: ImageTk.PhotoImage | None = None
        self.refined_photo: ImageTk.PhotoImage | None = None
        self.original_array: np.ndarray | None = None
        self.raw_array: np.ndarray | None = None
        self.mask_array: np.ndarray | None = None
        self.current_path: Path | None = None
        self.current_meta = ""
        self.source_paths: dict[str, Path] = {}
        self.source_var = tk.StringVar(value="image_path")
        self.source_combo: ttk.Combobox | None = None
        self.render_after_id: str | None = None
        self.active_filter_pipeline: dict[str, object] | None = None
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.DoubleVar(value=0.0)
        self.cluster_k_var = tk.IntVar(value=4)
        self.directional_cancel_var = tk.BooleanVar(value=False)
        self.directional_strength_var = tk.DoubleVar(value=0.8)
        self.directional_radius_var = tk.IntVar(value=8)
        self.cluster_mask_only_var = tk.BooleanVar(value=True)
        self.remove_border_bg_var = tk.BooleanVar(value=False)
        self.refine_enable_var = tk.BooleanVar(value=True)
        self.refine_kernel_var = tk.IntVar(value=3)
        self.refine_open_iter_var = tk.IntVar(value=1)
        self.refine_close_iter_var = tk.IntVar(value=1)
        self.refine_min_area_var = tk.IntVar(value=12)
        self.overlap_var = tk.BooleanVar(value=False)
        self.refine_kernel_w_var = tk.IntVar(value=3)
        self.refine_kernel_h_var = tk.IntVar(value=3)
        self.refine_angle_var = tk.BooleanVar(value=False)
        self.refine_ratio_var = tk.BooleanVar(value=False)
        self.jbf_enable_var = tk.BooleanVar(value=False)
        self.jbf_diameter_var = tk.IntVar(value=15)
        self.jbf_sigma_color_var = tk.DoubleVar(value=30.0)
        self.jbf_sigma_space_var = tk.DoubleVar(value=15.0)
        self.jbf_morph_open_var = tk.IntVar(value=3)
        self.jbf_morph_close_var = tk.IntVar(value=5)
        self.jbf_blur_kernel_var = tk.IntVar(value=3)
        self.jbf_threshold_var = tk.DoubleVar(value=230.0)

    def _ensure_window(self) -> None:
        if self.window is not None and self.window.winfo_exists():
            return

        self.window = tk.Toplevel(self.master)
        self.window.title("선택 패치 분석")
        self.window.geometry("1680x920")
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

        ttk.Label(control_frame, text="K 그룹(1=off)").pack(side=tk.LEFT, padx=(0, 6))
        k_scale = tk.Scale(
            control_frame,
            from_=1,
            to=8,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.cluster_k_var,
            length=190,
            command=lambda _value: self.schedule_render(),
        )
        k_scale.pack(side=tk.LEFT, padx=(0, 18))
        ttk.Button(control_frame, text="Reset", command=self.reset_controls).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(control_frame, text="K-Means는 보정/contrast/blur 적용 후 RGB 픽셀값 기준").pack(side=tk.LEFT)

        param_notebook = ttk.Notebook(self.window)
        param_notebook.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        kmeans_tab = ttk.Frame(param_notebook, padding=(8, 6))
        param_notebook.add(kmeans_tab, text="K-Means")
        refine_tab = ttk.Frame(param_notebook, padding=(8, 6))
        param_notebook.add(refine_tab, text="Refinement")
        direction_tab = ttk.Frame(param_notebook, padding=(8, 6))
        param_notebook.add(direction_tab, text="Directional")
        kmeans_row = ttk.Frame(kmeans_tab)
        kmeans_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Checkbutton(
            kmeans_row,
            text="Mask 영역 K-Means",
            variable=self.cluster_mask_only_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            kmeans_row,
            text="외곽 배경 제거",
            variable=self.remove_border_bg_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            kmeans_row,
            text="Overlap",
            variable=self.overlap_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(kmeans_row, text="현재 Filter 가져오기", command=self.apply_current_filter_pipeline).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(kmeans_row, text="외곽 배경 제거는 K>1에서 mask 외곽 band가 가장 닮은 cluster를 배경으로 제거").pack(side=tk.LEFT)

        direction_row = ttk.Frame(direction_tab)
        direction_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Checkbutton(
            direction_row,
            text="수직 상쇄",
            variable=self.directional_cancel_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(direction_row, text="강도").pack(side=tk.LEFT, padx=(0, 4))
        cancel_strength_scale = tk.Scale(
            direction_row,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.directional_strength_var,
            length=180,
            command=lambda _value: self.schedule_render(),
        )
        cancel_strength_scale.pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(direction_row, text="탐색").pack(side=tk.LEFT, padx=(0, 4))
        cancel_radius_scale = tk.Scale(
            direction_row,
            from_=2,
            to=24,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.directional_radius_var,
            length=180,
            command=lambda _value: self.schedule_render(),
        )
        cancel_radius_scale.pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(direction_row, text="mask 방향과 수직인 주변 픽셀로 선형 성분을 상쇄").pack(side=tk.LEFT)

        basic_refine_row = ttk.Frame(refine_tab)
        basic_refine_row.pack(side=tk.TOP, fill=tk.X)
        jbf_refine_row = ttk.Frame(refine_tab)
        jbf_refine_row.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        ttk.Checkbutton(
            basic_refine_row,
            text="Refinement 적용",
            variable=self.refine_enable_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            basic_refine_row,
            text="Angle",
            variable=self.refine_angle_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            basic_refine_row,
            text="Ratio",
            variable=self.refine_ratio_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="Kernel").pack(side=tk.LEFT, padx=(0, 4))
        refine_kernel_scale = tk.Scale(
            basic_refine_row,
            from_=1,
            to=9,
            resolution=2,
            orient=tk.HORIZONTAL,
            variable=self.refine_kernel_var,
            length=150,
            command=lambda _value: self.schedule_render(),
        )
        refine_kernel_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="W").pack(side=tk.LEFT, padx=(0, 4))
        refine_kernel_w_scale = tk.Scale(
            basic_refine_row,
            from_=1,
            to=31,
            resolution=2,
            orient=tk.HORIZONTAL,
            variable=self.refine_kernel_w_var,
            length=150,
            command=lambda _value: self.schedule_render(),
        )
        refine_kernel_w_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="H").pack(side=tk.LEFT, padx=(0, 4))
        refine_kernel_h_scale = tk.Scale(
            basic_refine_row,
            from_=1,
            to=31,
            resolution=2,
            orient=tk.HORIZONTAL,
            variable=self.refine_kernel_h_var,
            length=150,
            command=lambda _value: self.schedule_render(),
        )
        refine_kernel_h_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="Open").pack(side=tk.LEFT, padx=(0, 4))
        refine_open_scale = tk.Scale(
            basic_refine_row,
            from_=0,
            to=4,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.refine_open_iter_var,
            length=120,
            command=lambda _value: self.schedule_render(),
        )
        refine_open_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="Close").pack(side=tk.LEFT, padx=(0, 4))
        refine_close_scale = tk.Scale(
            basic_refine_row,
            from_=0,
            to=4,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.refine_close_iter_var,
            length=120,
            command=lambda _value: self.schedule_render(),
        )
        refine_close_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="Min area").pack(side=tk.LEFT, padx=(0, 4))
        refine_area_scale = tk.Scale(
            basic_refine_row,
            from_=0,
            to=300,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.refine_min_area_var,
            length=180,
            command=lambda _value: self.schedule_render(),
        )
        refine_area_scale.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(basic_refine_row, text="각 K-Means cluster별 clean 후 작은 component 제거").pack(side=tk.LEFT)

        ttk.Checkbutton(
            jbf_refine_row,
            text="JBFFilter",
            variable=self.jbf_enable_var,
            command=self.schedule_render,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(jbf_refine_row, text="Diameter").pack(side=tk.LEFT, padx=(0, 4))
        jbf_diameter_scale = tk.Scale(
            jbf_refine_row,
            from_=1,
            to=31,
            resolution=2,
            orient=tk.HORIZONTAL,
            variable=self.jbf_diameter_var,
            length=115,
            command=lambda _value: self.schedule_render(),
        )
        jbf_diameter_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="SigmaColor").pack(side=tk.LEFT, padx=(0, 4))
        jbf_sigma_color_scale = tk.Scale(
            jbf_refine_row,
            from_=1.0,
            to=150.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            variable=self.jbf_sigma_color_var,
            length=120,
            command=lambda _value: self.schedule_render(),
        )
        jbf_sigma_color_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="SigmaSpace").pack(side=tk.LEFT, padx=(0, 4))
        jbf_sigma_space_scale = tk.Scale(
            jbf_refine_row,
            from_=1.0,
            to=50.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            variable=self.jbf_sigma_space_var,
            length=120,
            command=lambda _value: self.schedule_render(),
        )
        jbf_sigma_space_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="MorphOpen").pack(side=tk.LEFT, padx=(0, 4))
        jbf_open_scale = tk.Scale(
            jbf_refine_row,
            from_=0,
            to=5,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.jbf_morph_open_var,
            length=90,
            command=lambda _value: self.schedule_render(),
        )
        jbf_open_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="MorphClose").pack(side=tk.LEFT, padx=(0, 4))
        jbf_close_scale = tk.Scale(
            jbf_refine_row,
            from_=0,
            to=5,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.jbf_morph_close_var,
            length=90,
            command=lambda _value: self.schedule_render(),
        )
        jbf_close_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="BlurKernel").pack(side=tk.LEFT, padx=(0, 4))
        jbf_blur_scale = tk.Scale(
            jbf_refine_row,
            from_=1,
            to=31,
            resolution=2,
            orient=tk.HORIZONTAL,
            variable=self.jbf_blur_kernel_var,
            length=105,
            command=lambda _value: self.schedule_render(),
        )
        jbf_blur_scale.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(jbf_refine_row, text="Threshold(0-255)").pack(side=tk.LEFT, padx=(0, 4))
        jbf_threshold_scale = tk.Scale(
            jbf_refine_row,
            from_=0.0,
            to=255.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            variable=self.jbf_threshold_var,
            length=120,
            command=lambda _value: self.schedule_render(),
        )
        jbf_threshold_scale.pack(side=tk.LEFT, padx=(0, 8))

        view_frame = ttk.Frame(self.window)
        view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)
        view_frame.columnconfigure(2, weight=1)
        view_frame.rowconfigure(1, weight=1)

        ttk.Label(view_frame, text="Contrast 조정 이미지", anchor="center").grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Label(view_frame, text="K-Means 그룹 결과", anchor="center").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(view_frame, text="Refinement 결과", anchor="center").grid(row=0, column=2, sticky="ew", padx=4)

        self.adjusted_image_label = ttk.Label(view_frame, anchor="center")
        self.adjusted_image_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.cluster_image_label = ttk.Label(view_frame, anchor="center")
        self.cluster_image_label.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.refined_image_label = ttk.Label(view_frame, anchor="center")
        self.refined_image_label.grid(row=1, column=2, sticky="nsew", padx=4, pady=4)

        self.cluster_stats_label = ttk.Label(self.window, text="", anchor="w", justify="left", wraplength=1500)
        self.cluster_stats_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

    def show_error(self, message: str) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None
        self.meta_label.config(text=message)
        if self.adjusted_image_label is not None:
            self.adjusted_image_label.config(image="", text="이미지를 불러오지 못했습니다.")
        if self.cluster_image_label is not None:
            self.cluster_image_label.config(image="", text="")
        if self.refined_image_label is not None:
            self.refined_image_label.config(image="", text="")
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
        self.remove_border_bg_var.set(False)
        self.refine_enable_var.set(True)
        self.refine_angle_var.set(False)
        self.refine_ratio_var.set(False)
        self.refine_kernel_var.set(3)
        self.refine_kernel_w_var.set(3)
        self.refine_kernel_h_var.set(3)
        self.refine_open_iter_var.set(1)
        self.refine_close_iter_var.set(1)
        self.refine_min_area_var.set(12)
        self.jbf_enable_var.set(False)
        self.jbf_diameter_var.set(15)
        self.jbf_sigma_color_var.set(30.0)
        self.jbf_sigma_space_var.set(15.0)
        self.jbf_morph_open_var.set(3)
        self.jbf_morph_close_var.set(5)
        self.jbf_blur_kernel_var.set(3)
        self.jbf_threshold_var.set(230.0)
        self.overlap_var.set(False)
        self.active_filter_pipeline = None
        self.render_images()

    def apply_current_filter_pipeline(self) -> None:
        if self.filter_pipeline_provider is None:
            messagebox.showwarning("Filter 없음", "메인 화면의 Filter Pipeline을 가져올 수 없습니다.")
            return
        pipeline = deepcopy(self.filter_pipeline_provider())
        self.active_filter_pipeline = pipeline
        source_name = str(pipeline.get("source", self.source_var.get()))
        source_changed = False
        if source_name in self.source_paths:
            source_changed = self.source_var.get() != source_name
            self.source_var.set(source_name)

        self.contrast_var.set(float(pipeline.get("contrast", 1.0)))
        self.blur_var.set(float(pipeline.get("blur", 0.0)))
        self.cluster_k_var.set(int(pipeline.get("k", 1)) if bool(pipeline.get("kmeans_enabled", False)) else 1)
        self.directional_cancel_var.set(False)
        self.cluster_mask_only_var.set(True)
        self.remove_border_bg_var.set(str(pipeline.get("cluster_select", "")) == "border_background")
        self.refine_enable_var.set(bool(pipeline.get("refine_enabled", False)))
        self.refine_angle_var.set(bool(pipeline.get("refine_angle_enabled", False)))
        self.refine_kernel_var.set(int(pipeline.get("refine_kernel", 3)))
        self.refine_kernel_w_var.set(int(pipeline.get("refine_kernel_w", 3)))
        self.refine_kernel_h_var.set(int(pipeline.get("refine_kernel_h", 3)))
        self.refine_open_iter_var.set(int(pipeline.get("refine_open_iter", 1)))
        self.refine_close_iter_var.set(int(pipeline.get("refine_close_iter", 1)))
        self.refine_min_area_var.set(int(pipeline.get("refine_min_area", 12)))
        self.jbf_enable_var.set(bool(pipeline.get("jbf_enabled", False)))
        self.jbf_diameter_var.set(int(pipeline.get("jbf_diameter", 15)))
        self.jbf_sigma_color_var.set(float(pipeline.get("jbf_sigma_color", 30.0)))
        self.jbf_sigma_space_var.set(float(pipeline.get("jbf_sigma_space", 15.0)))
        self.jbf_morph_open_var.set(int(pipeline.get("jbf_morph_open", 3)))
        self.jbf_morph_close_var.set(int(pipeline.get("jbf_morph_close", 5)))
        self.jbf_blur_kernel_var.set(int(pipeline.get("jbf_blur_kernel", 3)))
        self.jbf_threshold_var.set(float(pipeline.get("jbf_threshold", 230.0)))
        if source_changed and self.original_array is not None:
            self.load_selected_source()
        else:
            self.render_images()

    def active_pipeline_from_controls(self) -> dict[str, object]:
        pipeline = deepcopy(self.active_filter_pipeline) if self.active_filter_pipeline is not None else {}
        existing_cluster_select = str(pipeline.get("cluster_select", "all"))
        if bool(self.remove_border_bg_var.get()) and int(self.cluster_k_var.get()) > 1:
            cluster_select = "border_background"
        elif existing_cluster_select == "border_background":
            cluster_select = "all"
        else:
            cluster_select = existing_cluster_select
        pipeline.update(
            {
                "source": self.source_var.get(),
                "bbox_gate_enabled": bool(pipeline.get("bbox_gate_enabled", False)),
                "bbox_gate_min_ratio": float(pipeline.get("bbox_gate_min_ratio", 10.0)),
                "contrast": float(self.contrast_var.get()),
                "blur": float(self.blur_var.get()),
                "kmeans_enabled": int(self.cluster_k_var.get()) > 1,
                "k": int(self.cluster_k_var.get()),
                "cluster_select": cluster_select,
                "refine_enabled": bool(self.refine_enable_var.get()),
                "refine_kernel": int(self.refine_kernel_var.get()),
                "refine_kernel_w": int(self.refine_kernel_w_var.get()),
                "refine_kernel_h": int(self.refine_kernel_h_var.get()),
                "refine_angle_enabled": bool(self.refine_angle_var.get()),
                "refine_open_iter": int(self.refine_open_iter_var.get()),
                "refine_close_iter": int(self.refine_close_iter_var.get()),
                "refine_min_area": int(self.refine_min_area_var.get()),
                "jbf_enabled": bool(self.jbf_enable_var.get()),
                "jbf_diameter": int(self.jbf_diameter_var.get()),
                "jbf_sigma_color": float(self.jbf_sigma_color_var.get()),
                "jbf_sigma_space": float(self.jbf_sigma_space_var.get()),
                "jbf_morph_open": int(self.jbf_morph_open_var.get()),
                "jbf_morph_close": int(self.jbf_morph_close_var.get()),
                "jbf_blur_kernel": int(self.jbf_blur_kernel_var.get()),
                "jbf_threshold": float(self.jbf_threshold_var.get()),
            }
        )
        pipeline.setdefault("name", "Patch Filter Preview")
        pipeline.setdefault("cluster_select", "all")
        pipeline.setdefault("post_order", "jbf_then_refine")
        return pipeline

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
        assert self.adjusted_image_label is not None and self.cluster_image_label is not None and self.refined_image_label is not None

        contrast = float(self.contrast_var.get())
        blur_radius = float(self.blur_var.get())
        k = int(self.cluster_k_var.get())
        base, correction_text = self.make_analysis_base()
        angle_enabled = bool(self.refine_angle_var.get())
        ratio_enabled = bool(self.refine_ratio_var.get())
        if self.active_filter_pipeline is not None and self.pipeline_preview_renderer is not None:
            pipeline = self.active_pipeline_from_controls()
            adjusted, clustered, refined, stats_text = self.pipeline_preview_renderer(base, self.mask_array, pipeline)
            angle_enabled = bool(pipeline.get("refine_angle_enabled", False))
            stats_text = f"pipeline={pipeline.get('name', '')} | {stats_text}"
        else:
            adjusted = self.apply_blur(self.apply_contrast(base, contrast), blur_radius)
            cluster_mask = self.mask_array if self.cluster_mask_only_var.get() else None
            clustered, refined, stats_text = self.kmeans_cluster_image(
                adjusted,
                k,
                mask=cluster_mask,
                refine_enabled=bool(self.refine_enable_var.get()),
                refine_kernel=int(self.refine_kernel_var.get()),
                refine_kernel_w=int(self.refine_kernel_w_var.get()),
                refine_kernel_h=int(self.refine_kernel_h_var.get()),
                refine_angle_enabled=angle_enabled,
                angle_mask=self.mask_array,
                jbf_enabled=bool(self.jbf_enable_var.get()),
                jbf_diameter=int(self.jbf_diameter_var.get()),
                jbf_sigma_color=float(self.jbf_sigma_color_var.get()),
                jbf_sigma_space=float(self.jbf_sigma_space_var.get()),
                jbf_morph_open=int(self.jbf_morph_open_var.get()),
                jbf_morph_close=int(self.jbf_morph_close_var.get()),
                jbf_blur_kernel=int(self.jbf_blur_kernel_var.get()),
                jbf_threshold=float(self.jbf_threshold_var.get()),
                refine_open_iter=int(self.refine_open_iter_var.get()),
                refine_close_iter=int(self.refine_close_iter_var.get()),
                refine_min_area=int(self.refine_min_area_var.get()),
                remove_border_background=bool(self.remove_border_bg_var.get()),
            )
        if self.overlap_var.get():
            clustered_display = self.overlay_result(adjusted, clustered, alpha=0.8)
            refined_display = self.overlay_result(adjusted, refined, alpha=0.8)
        else:
            clustered_display = clustered
            refined_display = refined
        if ratio_enabled and self.mask_array is not None:
            refined_display = self.draw_angle_overlay(
                refined_display,
                self.mask_array,
                show_angle=False,
                show_ratio=True,
            )

        self.adjusted_photo = ImageTk.PhotoImage(self.to_display_image(adjusted))
        self.cluster_photo = ImageTk.PhotoImage(self.to_display_image(clustered_display))
        self.refined_photo = ImageTk.PhotoImage(self.to_display_image(refined_display))
        self.adjusted_image_label.config(image=self.adjusted_photo, text="")
        self.cluster_image_label.config(image=self.cluster_photo, text="")
        self.refined_image_label.config(image=self.refined_photo, text="")
        if self.cluster_stats_label is not None:
            self.cluster_stats_label.config(
                text=f"contrast={contrast:.2f}, blur={blur_radius:.1f}, K={k}, mask_only={self.cluster_mask_only_var.get()}, overlap={self.overlap_var.get()} | {correction_text} | {stats_text}"
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
    def overlay_result(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.8) -> np.ndarray:
        base_f = base.astype(np.float32)
        overlay_f = overlay.astype(np.float32)
        overlay_mask = np.any(overlay > 0, axis=2)
        out = base_f.copy()
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out[overlay_mask] = (1.0 - alpha) * base_f[overlay_mask] + alpha * overlay_f[overlay_mask]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_readable_text(
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        lines: list[str],
        fill: tuple[int, int, int],
        bg_fill: tuple[int, int, int] = (0, 0, 0),
        padding: int = 3,
    ) -> None:
        if not lines:
            return
        x, y = xy
        text = "\n".join(lines)
        try:
            bbox = draw.multiline_textbbox((x, y), text, spacing=2)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            widths = [draw.textlength(line) for line in lines]
            text_w = int(max(widths, default=0))
            text_h = 12 * len(lines) + 2 * max(len(lines) - 1, 0)
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=bg_fill,
        )
        draw.multiline_text((x, y), text, fill=fill, spacing=2)

    @classmethod
    def draw_angle_overlay(
        cls,
        array: np.ndarray,
        mask: np.ndarray,
        max_components: int = 80,
        show_angle: bool = True,
        show_ratio: bool = False,
    ) -> np.ndarray:
        if mask is None or mask.shape != array.shape[:2] or not mask.any():
            return array

        image = Image.fromarray(array.astype(np.uint8).copy(), mode="RGB")
        draw = ImageDraw.Draw(image)
        h, w = mask.shape
        line_width = max(2, int(round(min(h, w) / 320)))
        components = cls.connected_components_bool(mask, min_area=8)
        components.sort(key=len, reverse=True)

        for index, coords in enumerate(components[:max_components]):
            geometry = cls.component_angle_geometry(coords)
            corners = np.asarray(geometry["corners_xy"], dtype=np.float32)
            angle = float(geometry["angle_deg"])
            center_x, center_y = (float(v) for v in geometry["center_xy"])
            major_len = max(float(geometry["major_length"]), 12.0)
            minor_len = max(float(geometry["minor_length"]), 12.0)
            angle_rad = math.radians(angle)
            ux, uy = math.cos(angle_rad), math.sin(angle_rad)
            vx, vy = -uy, ux

            polygon = [(float(x), float(y)) for x, y in corners]
            draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=line_width)

            if show_angle:
                w_axis = min(max(major_len * 0.5, 18.0), 90.0)
                h_axis = min(max(minor_len * 0.5, 14.0), 70.0)
                w0 = (center_x - ux * w_axis, center_y - uy * w_axis)
                w1 = (center_x + ux * w_axis, center_y + uy * w_axis)
                h0 = (center_x - vx * h_axis, center_y - vy * h_axis)
                h1 = (center_x + vx * h_axis, center_y + vy * h_axis)

                draw.line([w0, w1], fill=(255, 255, 0), width=line_width + 1)
                draw.line([h0, h1], fill=(0, 255, 255), width=line_width + 1)
                draw.ellipse(
                    [center_x - line_width, center_y - line_width, center_x + line_width, center_y + line_width],
                    fill=(255, 0, 0),
                )

            if index < 20:
                label_lines = []
                if show_angle:
                    label_lines.append(f"A={angle:.1f} deg")
                if show_ratio:
                    ratio = major_len / max(minor_len, 1e-6)
                    label_lines.append(f"BBox {major_len:.0f}x{minor_len:.0f}")
                    label_lines.append(f"R={ratio:.2f}")
                label_x = max(0.0, min(w - 120.0, min(x for x, _y in polygon)))
                label_y = max(0.0, min(h - 46.0, min(y for _x, y in polygon) - 36.0))
                cls.draw_readable_text(draw, (label_x, label_y), label_lines, fill=(255, 255, 255))
                if show_angle:
                    draw.text((w1[0] + 2, w1[1] + 2), "W", fill=(255, 255, 0))
                    draw.text((h1[0] + 2, h1[1] + 2), "H", fill=(0, 255, 255))

        return np.asarray(image, dtype=np.uint8)

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
    def normalize_axis_angle_deg(angle: float) -> float:
        angle = float(angle)
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return angle

    @classmethod
    def component_angle_geometry(cls, coords_yx: np.ndarray) -> dict[str, object]:
        coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
        if len(coords_xy) == 0:
            return {
                "angle_deg": 0.0,
                "center_xy": (0.0, 0.0),
                "corners_xy": np.zeros((4, 2), dtype=np.float32),
                "major_length": 0.0,
                "minor_length": 0.0,
            }

        if len(coords_xy) < 2:
            x, y = coords_xy[0]
            corners = np.array([[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]], dtype=np.float32)
            return {
                "angle_deg": 0.0,
                "center_xy": (float(x), float(y)),
                "corners_xy": corners,
                "major_length": 1.0,
                "minor_length": 1.0,
            }

        if cv2 is not None and len(coords_xy) >= 5:
            rect = cv2.minAreaRect(coords_xy)
            width, height = rect[1]
            angle = float(rect[2])
            major_length = float(width)
            minor_length = float(height)
            if width < height:
                angle += 90.0
                major_length, minor_length = float(height), float(width)
            return {
                "angle_deg": cls.normalize_axis_angle_deg(angle),
                "center_xy": (float(rect[0][0]), float(rect[0][1])),
                "corners_xy": cv2.boxPoints(rect).astype(np.float32),
                "major_length": major_length,
                "minor_length": minor_length,
            }

        centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, int(np.argmax(eigvals))]
        angle = cls.normalize_axis_angle_deg(math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))))
        angle_rad = math.radians(angle)
        major_axis = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        minor_axis = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)
        major_projection = coords_xy @ major_axis
        minor_projection = coords_xy @ minor_axis
        major_min, major_max = float(major_projection.min()), float(major_projection.max())
        minor_min, minor_max = float(minor_projection.min()), float(minor_projection.max())
        corners = np.array(
            [
                major_axis * major_min + minor_axis * minor_min,
                major_axis * major_max + minor_axis * minor_min,
                major_axis * major_max + minor_axis * minor_max,
                major_axis * major_min + minor_axis * minor_max,
            ],
            dtype=np.float32,
        )
        return {
            "angle_deg": angle,
            "center_xy": (float(coords_xy[:, 0].mean()), float(coords_xy[:, 1].mean())),
            "corners_xy": corners,
            "major_length": max(major_max - major_min, 1.0),
            "minor_length": max(minor_max - minor_min, 1.0),
        }

    @classmethod
    def component_major_angle_deg(cls, coords_yx: np.ndarray) -> float:
        return float(cls.component_angle_geometry(coords_yx)["angle_deg"])

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
    def binary_dilate(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return out

    @staticmethod
    def binary_erode(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        out = np.ones_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return out

    @staticmethod
    def binary_dilate_rect(mask: np.ndarray, kernel_w: int, kernel_h: int) -> np.ndarray:
        mask_bool = mask.astype(bool)
        rx = max(int(kernel_w) // 2, 0)
        ry = max(int(kernel_h) // 2, 0)
        padded_x = np.pad(mask_bool, ((0, 0), (rx, rx)), mode="constant", constant_values=False)
        horiz = np.zeros_like(mask_bool, dtype=bool)
        for dx in range(2 * rx + 1):
            horiz |= padded_x[:, dx : dx + mask_bool.shape[1]]
        padded_y = np.pad(horiz, ((ry, ry), (0, 0)), mode="constant", constant_values=False)
        out = np.zeros_like(mask_bool, dtype=bool)
        for dy in range(2 * ry + 1):
            out |= padded_y[dy : dy + mask_bool.shape[0], :]
        return out

    @staticmethod
    def binary_erode_rect(mask: np.ndarray, kernel_w: int, kernel_h: int) -> np.ndarray:
        mask_bool = mask.astype(bool)
        rx = max(int(kernel_w) // 2, 0)
        ry = max(int(kernel_h) // 2, 0)
        padded_x = np.pad(mask_bool, ((0, 0), (rx, rx)), mode="constant", constant_values=False)
        horiz = np.ones_like(mask_bool, dtype=bool)
        for dx in range(2 * rx + 1):
            horiz &= padded_x[:, dx : dx + mask_bool.shape[1]]
        padded_y = np.pad(horiz, ((ry, ry), (0, 0)), mode="constant", constant_values=False)
        out = np.ones_like(mask_bool, dtype=bool)
        for dy in range(2 * ry + 1):
            out &= padded_y[dy : dy + mask_bool.shape[0], :]
        return out

    @classmethod
    def morphology_clean(
        cls,
        mask: np.ndarray,
        kernel_size: int = 3,
        kernel_w: int | None = None,
        kernel_h: int | None = None,
        open_iter: int = 1,
        close_iter: int = 1,
    ) -> np.ndarray:
        mask_bool = mask.astype(bool)
        kernel_size = int(np.clip(kernel_size, 1, 31))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_w is None:
            kernel_w = kernel_size
        if kernel_h is None:
            kernel_h = kernel_size
        kernel_w = int(np.clip(kernel_w, 1, 31))
        kernel_h = int(np.clip(kernel_h, 1, 31))
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        open_iter = int(np.clip(open_iter, 0, 10))
        close_iter = int(np.clip(close_iter, 0, 10))
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
            as_u8 = mask_bool.astype(np.uint8)
            opened = cv2.morphologyEx(as_u8, cv2.MORPH_OPEN, kernel, iterations=open_iter) if open_iter else as_u8
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter) if close_iter else opened
            return closed.astype(bool)

        opened = mask_bool
        for _ in range(open_iter):
            opened = cls.binary_dilate_rect(cls.binary_erode_rect(opened, kernel_w, kernel_h), kernel_w, kernel_h)
        closed = opened
        for _ in range(close_iter):
            closed = cls.binary_erode_rect(cls.binary_dilate_rect(closed, kernel_w, kernel_h), kernel_w, kernel_h)
        return closed

    @classmethod
    def remove_small_components(cls, mask: np.ndarray, min_area: int) -> np.ndarray:
        mask_bool = mask.astype(bool)
        min_area = int(max(min_area, 0))
        if min_area <= 1 or not mask_bool.any():
            return mask_bool

        if cv2 is not None:
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
            keep = np.zeros(num_labels, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
            return keep[labels]

        out = np.zeros_like(mask_bool, dtype=bool)
        for coords in cls.connected_components_bool(mask_bool, min_area=min_area):
            out[coords[:, 0], coords[:, 1]] = True
        return out

    @staticmethod
    def rotate_binary_patch(mask: np.ndarray, angle_deg: float) -> np.ndarray:
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        rotated = image.rotate(
            angle=float(angle_deg),
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )
        return np.asarray(rotated, dtype=np.uint8) > 0

    @classmethod
    def morphology_clean_angle_aligned(
        cls,
        cluster_mask: np.ndarray,
        angle_mask: np.ndarray,
        kernel_size: int = 3,
        kernel_w: int | None = None,
        kernel_h: int | None = None,
        open_iter: int = 1,
        close_iter: int = 1,
    ) -> tuple[np.ndarray, int]:
        cluster_bool = cluster_mask.astype(bool)
        angle_bool = angle_mask.astype(bool)
        if not angle_bool.any():
            angle_bool = cluster_bool

        kernel_size = int(np.clip(kernel_size, 1, 31))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_w is None:
            kernel_w = kernel_size
        if kernel_h is None:
            kernel_h = kernel_size
        kernel_w = int(np.clip(kernel_w, 1, 31))
        kernel_h = int(np.clip(kernel_h, 1, 31))
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        open_iter = int(np.clip(open_iter, 0, 10))
        close_iter = int(np.clip(close_iter, 0, 10))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)) if cv2 is not None else None

        out = np.zeros_like(cluster_bool, dtype=bool)
        h, w = cluster_bool.shape
        pad = max(kernel_w, kernel_h, 5) * 2
        components = cls.connected_components_bool(angle_bool, min_area=8)
        if not components:
            components = cls.connected_components_bool(cluster_bool, min_area=1)

        angle_component_count = 0
        for coords in components:
            yy = coords[:, 0]
            xx = coords[:, 1]
            y0 = max(int(yy.min()) - pad, 0)
            y1 = min(int(yy.max()) + pad + 1, h)
            x0 = max(int(xx.min()) - pad, 0)
            x1 = min(int(xx.max()) + pad + 1, w)

            patch_cluster = cluster_bool[y0:y1, x0:x1]
            component_patch = np.zeros_like(patch_cluster, dtype=bool)
            component_patch[yy - y0, xx - x0] = True
            patch_input = patch_cluster & component_patch
            if not patch_input.any():
                continue

            angle = cls.component_major_angle_deg(coords)
            patch_h, patch_w = patch_input.shape
            center = ((patch_w - 1) / 2.0, (patch_h - 1) / 2.0)

            if cv2 is not None:
                rotate_to_axis = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    patch_input.astype(np.uint8),
                    rotate_to_axis,
                    (patch_w, patch_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if open_iter:
                    rotated = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel, iterations=open_iter)
                if close_iter:
                    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

                rotate_back = cv2.getRotationMatrix2D(center, -angle, 1.0)
                restored = cv2.warpAffine(
                    rotated,
                    rotate_back,
                    (patch_w, patch_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
            else:
                rotated = cls.rotate_binary_patch(patch_input, angle)
                cleaned = cls.morphology_clean(
                    rotated,
                    kernel_size=kernel_size,
                    kernel_w=kernel_w,
                    kernel_h=kernel_h,
                    open_iter=open_iter,
                    close_iter=close_iter,
                )
                restored = cls.rotate_binary_patch(cleaned, -angle)
            out[y0:y1, x0:x1] |= restored & component_patch
            angle_component_count += 1

        return out, angle_component_count

    @staticmethod
    def normalize_odd_kernel(value: int, minimum: int = 1, maximum: int = 31) -> int:
        kernel = int(np.clip(int(value), minimum, maximum))
        if kernel % 2 == 0:
            kernel += 1
        return int(np.clip(kernel, minimum, maximum))

    @staticmethod
    def blur_soft_mask(mask_float: np.ndarray, kernel_size: int) -> np.ndarray:
        kernel_size = ImageViewer.normalize_odd_kernel(kernel_size, minimum=1, maximum=31)
        if kernel_size <= 1:
            return mask_float.astype(np.float32)
        if cv2 is not None:
            return cv2.GaussianBlur(mask_float.astype(np.float32), (kernel_size, kernel_size), 0)
        radius = max((kernel_size - 1) / 6.0, 0.1)
        image = Image.fromarray(np.clip(mask_float * 255.0, 0, 255).astype(np.uint8), mode="L")
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0

    @staticmethod
    def custom_joint_bilateral_filter(
        source_mask: np.ndarray,
        guide_array: np.ndarray,
        valid_mask: np.ndarray,
        diameter: int,
        sigma_color: float,
        sigma_space: float,
    ) -> np.ndarray:
        radius = max(diameter // 2, 0)
        if radius <= 0:
            return source_mask.astype(np.float32)

        source = source_mask.astype(np.float32)
        guide = guide_array.astype(np.float32)
        valid = valid_mask.astype(np.float32)
        sigma_color = max(float(sigma_color), 1e-3)
        sigma_space = max(float(sigma_space), 1e-3)
        padded_source = np.pad(source, ((radius, radius), (radius, radius)), mode="edge")
        padded_guide = np.pad(guide, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        padded_valid = np.pad(valid, ((radius, radius), (radius, radius)), mode="constant", constant_values=0.0)
        h, w = source.shape
        weighted_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted_source = padded_source[
                    radius + dy : radius + dy + h,
                    radius + dx : radius + dx + w,
                ]
                shifted_guide = padded_guide[
                    radius + dy : radius + dy + h,
                    radius + dx : radius + dx + w,
                    :,
                ]
                shifted_valid = padded_valid[
                    radius + dy : radius + dy + h,
                    radius + dx : radius + dx + w,
                ]
                spatial_term = (dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space)
                color_dist2 = ((guide - shifted_guide) ** 2).sum(axis=2)
                color_term = color_dist2 / (2.0 * sigma_color * sigma_color)
                weight = np.exp(-(spatial_term + color_term)).astype(np.float32) * shifted_valid
                weighted_sum += shifted_source * weight
                weight_sum += weight
        return weighted_sum / np.maximum(weight_sum, 1e-6)

    @staticmethod
    def has_opencv_joint_bilateral_filter() -> bool:
        ximgproc = getattr(cv2, "ximgproc", None) if cv2 is not None else None
        return callable(getattr(ximgproc, "jointBilateralFilter", None))

    @classmethod
    def jbf_refine_mask(
        cls,
        cluster_mask: np.ndarray,
        guide_array: np.ndarray | None,
        valid_mask: np.ndarray,
        diameter: int,
        sigma_color: float,
        sigma_space: float,
        morph_open: int,
        morph_close: int,
        blur_kernel: int,
        threshold: float,
        require_opencv_jbf: bool = False,
    ) -> tuple[np.ndarray, float, str]:
        start = time.perf_counter()
        cluster_bool = cluster_mask.astype(bool)
        if guide_array is None or not cluster_bool.any():
            return cluster_bool, (time.perf_counter() - start) * 1000, "skip"

        valid_bool = valid_mask.astype(bool)
        roi_mask = valid_bool | cluster_bool
        yy, xx = np.where(roi_mask)
        if len(yy) == 0:
            return cluster_bool, (time.perf_counter() - start) * 1000, "empty"

        diameter = cls.normalize_odd_kernel(diameter, minimum=1, maximum=31)
        blur_kernel = cls.normalize_odd_kernel(blur_kernel, minimum=1, maximum=31) if int(blur_kernel) > 0 else 0
        sigma_color = max(float(sigma_color), 1e-3)
        sigma_space = max(float(sigma_space), 1e-3)
        threshold_value = float(np.clip(threshold, 0.0, 255.0))
        threshold_unit = threshold_value / 255.0
        morph_open = int(np.clip(morph_open, 0, 10))
        morph_close = int(np.clip(morph_close, 0, 10))

        h, w = cluster_bool.shape
        pad = max(diameter // 2, blur_kernel // 2, 2)
        y0 = max(int(yy.min()) - pad, 0)
        y1 = min(int(yy.max()) + pad + 1, h)
        x0 = max(int(xx.min()) - pad, 0)
        x1 = min(int(xx.max()) + pad + 1, w)

        source_roi = cluster_bool[y0:y1, x0:x1].astype(np.float32)
        valid_roi = valid_bool[y0:y1, x0:x1]
        guide_roi = guide_array[y0:y1, x0:x1].astype(np.uint8)
        mode = "custom"
        filtered: np.ndarray | None = None

        if cls.has_opencv_joint_bilateral_filter():
            try:
                source_u8 = np.clip(source_roi * 255.0, 0, 255).astype(np.uint8)
                filtered_u8 = cv2.ximgproc.jointBilateralFilter(
                    guide_roi,
                    source_u8,
                    diameter,
                    sigma_color,
                    sigma_space,
                )
                filtered = filtered_u8.astype(np.float32) / 255.0
                mode = "opencv-jbf"
            except Exception:  # noqa: BLE001 - optional OpenCV contrib API varies by build
                filtered = None
                if require_opencv_jbf:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return cluster_bool, elapsed_ms, "opencv-jbf-error"
        elif require_opencv_jbf:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return cluster_bool, elapsed_ms, "opencv-jbf-unavailable"

        cost_estimate = int(source_roi.size) * diameter * diameter
        if filtered is None and cost_estimate <= 35_000_000:
            filtered = cls.custom_joint_bilateral_filter(
                source_roi,
                guide_roi,
                valid_roi,
                diameter=diameter,
                sigma_color=sigma_color,
                sigma_space=sigma_space,
            )
            mode = "custom"
        elif filtered is None and cv2 is not None:
            filtered = cv2.bilateralFilter(
                source_roi.astype(np.float32),
                diameter,
                max(sigma_color / 255.0, 1e-3),
                sigma_space,
            )
            mode = "mask-bilateral"
        elif filtered is None:
            filtered = cls.blur_soft_mask(source_roi, diameter)
            mode = "gaussian"

        if blur_kernel > 1:
            filtered = cls.blur_soft_mask(filtered, blur_kernel)
        clean_roi = (filtered >= threshold_unit) & valid_roi
        if morph_open or morph_close:
            clean_roi = cls.morphology_clean(
                clean_roi,
                kernel_size=3,
                kernel_w=3,
                kernel_h=3,
                open_iter=morph_open,
                close_iter=morph_close,
            )

        out = np.zeros_like(cluster_bool, dtype=bool)
        out[y0:y1, x0:x1] = clean_roi & valid_roi
        return out, (time.perf_counter() - start) * 1000, mode

    @classmethod
    def connected_component_summary(cls, mask: np.ndarray) -> dict[str, float]:
        clean = mask.astype(bool)
        clean_area = int(clean.sum())
        if clean_area <= 0:
            return {
                "component_count": 0.0,
                "largest_component_area": 0.0,
                "largest_component_ratio": 0.0,
                "small_component_ratio": 0.0,
                "largest_bbox_fill_ratio": 0.0,
            }

        if cv2 is not None:
            num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                return {
                    "component_count": 0.0,
                    "largest_component_area": 0.0,
                    "largest_component_ratio": 0.0,
                    "small_component_ratio": 0.0,
                    "largest_bbox_fill_ratio": 0.0,
                }
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
            widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
            heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        else:
            components = cls.connected_components_bool(clean, min_area=1)
            if not components:
                return {
                    "component_count": 0.0,
                    "largest_component_area": 0.0,
                    "largest_component_ratio": 0.0,
                    "small_component_ratio": 0.0,
                    "largest_bbox_fill_ratio": 0.0,
                }
            areas_list = []
            widths_list = []
            heights_list = []
            for coords in components:
                yy = coords[:, 0]
                xx = coords[:, 1]
                areas_list.append(float(len(coords)))
                widths_list.append(float(xx.max() - xx.min() + 1))
                heights_list.append(float(yy.max() - yy.min() + 1))
            areas = np.asarray(areas_list, dtype=np.float32)
            widths = np.asarray(widths_list, dtype=np.float32)
            heights = np.asarray(heights_list, dtype=np.float32)

        largest_idx = int(np.argmax(areas))
        largest_area = float(areas[largest_idx])
        bbox_area = float(max(widths[largest_idx] * heights[largest_idx], 1.0))
        small_threshold = max(8.0, clean_area * 0.02)
        small_area = float(areas[areas <= small_threshold].sum())
        return {
            "component_count": float(len(areas)),
            "largest_component_area": largest_area,
            "largest_component_ratio": largest_area / max(clean_area, 1),
            "small_component_ratio": small_area / max(clean_area, 1),
            "largest_bbox_fill_ratio": largest_area / bbox_area,
        }

    @classmethod
    def border_band_mask(cls, valid_mask: np.ndarray) -> np.ndarray:
        valid = valid_mask.astype(bool)
        if not valid.any():
            return valid
        eroded = cls.binary_erode_rect(valid, 3, 3)
        border = valid & ~eroded
        min_border = max(8, int(valid.sum() * 0.01))
        if int(border.sum()) < min_border:
            return valid
        return border

    @classmethod
    def infer_border_background_cluster(
        cls,
        labels_2d: np.ndarray,
        valid_mask: np.ndarray,
        centers: np.ndarray,
        image_array: np.ndarray,
        counts: np.ndarray | None = None,
    ) -> tuple[int | None, str]:
        if centers is None or len(centers) <= 1:
            return None, "border_bg=off"
        k = int(len(centers))
        border = cls.border_band_mask(valid_mask)
        border_labels = labels_2d[border]
        border_labels = border_labels[(border_labels >= 0) & (border_labels < k)]
        if len(border_labels) <= 0:
            if counts is not None and len(counts):
                cluster_id = int(np.argmax(counts))
                return cluster_id, f"border_bg=C{cluster_id}(fallback=largest)"
            return None, "border_bg=none"

        border_counts = np.bincount(border_labels.astype(np.int16), minlength=k).astype(np.float32)
        occupancy = border_counts / max(float(border_counts.sum()), 1.0)
        border_pixels = image_array[border].reshape(-1, 3).astype(np.float32)
        border_mean = border_pixels.mean(axis=0)
        center_dist = np.linalg.norm(centers.astype(np.float32) - border_mean[None, :], axis=1)
        dist_span = float(center_dist.max() - center_dist.min())
        if dist_span < 1e-6:
            color_score = np.ones(k, dtype=np.float32)
        else:
            color_score = 1.0 - ((center_dist - center_dist.min()) / (dist_span + 1e-6))
        score = occupancy * 0.75 + color_score * 0.25
        cluster_id = int(np.argmax(score))
        return (
            cluster_id,
            f"border_bg=C{cluster_id}, border_occ={occupancy[cluster_id]:.2f}, color_dist={center_dist[cluster_id]:.1f}",
        )

    @classmethod
    def cluster_spatial_summary(
        cls,
        labels_2d: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
        counts: np.ndarray,
        refine_kernel: int = 3,
        refine_kernel_w: int | None = None,
        refine_kernel_h: int | None = None,
        refine_enabled: bool = True,
        refine_angle_enabled: bool = False,
        angle_mask: np.ndarray | None = None,
        guide_array: np.ndarray | None = None,
        jbf_enabled: bool = False,
        jbf_diameter: int = 15,
        jbf_sigma_color: float = 30.0,
        jbf_sigma_space: float = 15.0,
        jbf_morph_open: int = 3,
        jbf_morph_close: int = 5,
        jbf_blur_kernel: int = 3,
        jbf_threshold: float = 230.0,
        refine_open_iter: int = 1,
        refine_close_iter: int = 1,
        refine_min_area: int = 12,
    ) -> tuple[str, float, list[dict[str, float]], np.ndarray]:
        start = time.perf_counter()
        total = max(int(valid_mask.sum()), 1)
        rows = []
        refined_labels = np.full(labels_2d.shape, -1, dtype=np.int16)
        angle_component_total = 0
        if angle_mask is not None and angle_mask.shape == valid_mask.shape and angle_mask.any():
            angle_ref_mask = angle_mask.astype(bool)
        else:
            angle_ref_mask = valid_mask
        jbf_elapsed_ms = 0.0
        jbf_modes: set[str] = set()
        for cluster_id in range(k):
            raw_cluster = (labels_2d == cluster_id) & valid_mask
            raw_area = int(raw_cluster.sum())
            if raw_area <= 0:
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "raw_area_ratio": 0.0,
                        "clean_survival_ratio": 0.0,
                        "component_count": 0.0,
                        "largest_component_ratio": 0.0,
                        "small_component_ratio": 0.0,
                        "largest_bbox_fill_ratio": 0.0,
                    }
                )
                continue

            work_cluster = raw_cluster
            if jbf_enabled:
                work_cluster, jbf_ms, jbf_mode = cls.jbf_refine_mask(
                    raw_cluster,
                    guide_array=guide_array,
                    valid_mask=valid_mask,
                    diameter=jbf_diameter,
                    sigma_color=jbf_sigma_color,
                    sigma_space=jbf_sigma_space,
                    morph_open=jbf_morph_open,
                    morph_close=jbf_morph_close,
                    blur_kernel=jbf_blur_kernel,
                    threshold=jbf_threshold,
                    require_opencv_jbf=True,
                )
                jbf_elapsed_ms += jbf_ms
                jbf_modes.add(jbf_mode)

            if refine_enabled:
                if refine_angle_enabled:
                    clean, angle_component_count = cls.morphology_clean_angle_aligned(
                        work_cluster,
                        angle_mask=angle_ref_mask,
                        kernel_size=refine_kernel,
                        kernel_w=refine_kernel_w,
                        kernel_h=refine_kernel_h,
                        open_iter=refine_open_iter,
                        close_iter=refine_close_iter,
                    )
                    angle_component_total += angle_component_count
                else:
                    clean = cls.morphology_clean(
                        work_cluster,
                        kernel_size=refine_kernel,
                        kernel_w=refine_kernel_w,
                        kernel_h=refine_kernel_h,
                        open_iter=refine_open_iter,
                        close_iter=refine_close_iter,
                    )
                clean = cls.remove_small_components(clean, min_area=refine_min_area)
            else:
                clean = work_cluster
            refined_labels[clean] = cluster_id
            clean_area = int(clean.sum())
            cc = cls.connected_component_summary(clean)
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "raw_area_ratio": raw_area / total,
                    "clean_survival_ratio": clean_area / max(raw_area, 1),
                    **cc,
                }
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        best = max(
            rows,
            key=lambda row: row["clean_survival_ratio"]
            * row["largest_component_ratio"]
            * math.log1p(row["raw_area_ratio"] * total),
        )
        compact = []
        for row in rows:
            compact.append(
                "C{cluster_id}: area={raw_area_ratio:.1%}, surv={clean_survival_ratio:.2f}, "
                "L={largest_component_ratio:.2f}, n={component_count:.0f}, small={small_component_ratio:.2f}, fill={largest_bbox_fill_ratio:.2f}".format(
                    **row
                )
            )
        angle_text = f", angle_components={angle_component_total}" if refine_enabled and refine_angle_enabled else ""
        jbf_text = ""
        if jbf_enabled:
            mode_text = ",".join(sorted(jbf_modes)) if jbf_modes else "off"
            jbf_text = f", jbf={jbf_elapsed_ms:.1f}ms[{mode_text}]"
        summary = f"spatial={elapsed_ms:.1f}ms, blob_like=C{int(best['cluster_id'])}{angle_text}{jbf_text} | " + " | ".join(compact)
        return summary, elapsed_ms, rows, refined_labels

    @staticmethod
    def kmeans_cluster_image(
        array: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
        refine_enabled: bool = True,
        refine_kernel: int = 3,
        refine_kernel_w: int | None = None,
        refine_kernel_h: int | None = None,
        refine_angle_enabled: bool = False,
        angle_mask: np.ndarray | None = None,
        jbf_enabled: bool = False,
        jbf_diameter: int = 15,
        jbf_sigma_color: float = 30.0,
        jbf_sigma_space: float = 15.0,
        jbf_morph_open: int = 3,
        jbf_morph_close: int = 5,
        jbf_blur_kernel: int = 3,
        jbf_threshold: float = 230.0,
        refine_open_iter: int = 1,
        refine_close_iter: int = 1,
        refine_min_area: int = 12,
        remove_border_background: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        kmeans_start = time.perf_counter()
        k = int(np.clip(k, 1, 8))
        h, w, _ = array.shape
        if mask is not None and mask.shape == (h, w) and mask.any():
            valid_mask = mask.astype(bool)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        pixels = array[valid_mask].reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(17)
        sample_size = min(25000, len(pixels))
        if sample_size <= 0:
            empty = np.zeros_like(array)
            return empty, empty, "no valid pixels"
        if k == 1:
            ordered_labels = np.zeros(len(pixels), dtype=np.int16)
            ordered_centers = pixels.mean(axis=0, keepdims=True).astype(np.float32)
            kmeans_mode_text = "off"
        else:
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
            kmeans_mode_text = f"{(time.perf_counter() - kmeans_start) * 1000:.1f}ms"

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
        if k == 1:
            clustered[valid_mask] = array[valid_mask]
        else:
            clustered[valid_mask] = palette[ordered_labels]
        counts = np.bincount(ordered_labels, minlength=k)
        total = max(int(counts.sum()), 1)
        labels_2d = np.full((h, w), -1, dtype=np.int16)
        labels_2d[valid_mask] = ordered_labels
        process_mask = valid_mask
        border_bg_text = "border_bg=off"
        if remove_border_background and k > 1:
            bg_cluster_id, border_bg_text = ImageViewer.infer_border_background_cluster(
                labels_2d,
                valid_mask,
                ordered_centers,
                array,
                counts,
            )
            if bg_cluster_id is not None:
                background_pixels = (labels_2d == bg_cluster_id) & valid_mask
                clustered[background_pixels] = 0
                process_mask = valid_mask & (labels_2d != bg_cluster_id)
        kmeans_ms = (time.perf_counter() - kmeans_start) * 1000
        spatial_text, _spatial_ms, _rows, refined_labels = ImageViewer.cluster_spatial_summary(
            labels_2d,
            process_mask,
            k,
            counts,
            refine_kernel=refine_kernel,
            refine_kernel_w=refine_kernel_w,
            refine_kernel_h=refine_kernel_h,
            refine_enabled=refine_enabled,
            refine_angle_enabled=refine_angle_enabled,
            angle_mask=angle_mask,
            guide_array=array,
            jbf_enabled=jbf_enabled,
            jbf_diameter=jbf_diameter,
            jbf_sigma_color=jbf_sigma_color,
            jbf_sigma_space=jbf_sigma_space,
            jbf_morph_open=jbf_morph_open,
            jbf_morph_close=jbf_morph_close,
            jbf_blur_kernel=jbf_blur_kernel,
            jbf_threshold=jbf_threshold,
            refine_open_iter=refine_open_iter,
            refine_close_iter=refine_close_iter,
            refine_min_area=refine_min_area,
        )
        if refine_enabled or jbf_enabled:
            refined = np.zeros((h, w, 3), dtype=np.uint8)
            refined_valid = refined_labels >= 0
            refined[refined_valid] = palette[refined_labels[refined_valid]]
        else:
            refined = clustered.copy()
        stats = []
        for cluster_id in range(k):
            mean_rgb = ordered_centers[cluster_id]
            rate = counts[cluster_id] / total * 100
            stats.append(
                f"C{cluster_id}: {rate:.1f}%, RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})"
            )
        refine_text = (
            f"refine={'on' if refine_enabled else 'off'}, angle={'on' if refine_angle_enabled else 'off'}, "
            f"kernel={refine_kernel}, kw={refine_kernel_w}, kh={refine_kernel_h}, "
            f"open={refine_open_iter}, close={refine_close_iter}, min_area={refine_min_area}, "
            f"JBF={'on' if jbf_enabled else 'off'}"
        )
        if jbf_enabled:
            refine_text += (
                f"(d={jbf_diameter}, sc={jbf_sigma_color:.1f}, ss={jbf_sigma_space:.1f}, "
                f"mo={jbf_morph_open}, mc={jbf_morph_close}, blur={jbf_blur_kernel}, th={jbf_threshold:.1f}/255)"
            )
        kmeans_text = f"kmeans={kmeans_mode_text}" if k == 1 else f"kmeans={kmeans_ms:.1f}ms"
        return clustered, refined, f"{kmeans_text} | {border_bg_text} | {refine_text} | color: {' | '.join(stats)} | {spatial_text}"


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
        self.main_filter_results: dict[int, dict[str, object]] = {}
        self.main_filter_cache: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
        self.main_filter_image_cache: OrderedDict[tuple[object, ...], np.ndarray | None] = OrderedDict()
        self.main_filter_mask_cache: OrderedDict[tuple[object, ...], np.ndarray | None] = OrderedDict()
        self.main_filter_result_cache_limit = 20000
        self.main_filter_image_cache_limit = 96
        self.main_filter_mask_cache_limit = 512
        self.main_filter_scope_text = ""
        self.filter_rate_ax = None
        self.filter_pipelines: list[dict[str, object]] = [self.default_filter_pipeline()]
        self.image_viewer = ImageViewer(
            root,
            filter_pipeline_provider=self.selected_filter_pipeline,
            pipeline_preview_renderer=self.render_filter_pipeline_preview,
        )

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
        self.main_filter_enable_var = tk.BooleanVar(value=False)
        self.main_filter_pipeline_var = tk.StringVar(value=str(self.filter_pipelines[0]["name"]))
        self.main_filter_sample_size_var = tk.StringVar(value="500")
        self.main_filter_seed_var = tk.StringVar(value="17")
        self.main_filter_group_balanced_var = tk.BooleanVar(value=False)
        self.main_filter_rate_line_var = tk.BooleanVar(value=False)
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

    @staticmethod
    def default_filter_pipeline(name: str = "OpenCV JBF Default") -> dict[str, object]:
        return {
            "name": name,
            "source": "image_path",
            "bbox_gate_enabled": False,
            "bbox_gate_min_ratio": 10.0,
            "contrast": 1.0,
            "blur": 0.0,
            "kmeans_enabled": False,
            "k": 1,
            "cluster_select": "all",
            "post_order": "jbf_then_refine",
            "refine_enabled": False,
            "refine_kernel": 3,
            "refine_kernel_w": 3,
            "refine_kernel_h": 3,
            "refine_angle_enabled": False,
            "refine_open_iter": 1,
            "refine_close_iter": 1,
            "refine_min_area": 12,
            "jbf_enabled": True,
            "jbf_diameter": 15,
            "jbf_sigma_color": 30.0,
            "jbf_sigma_space": 15.0,
            "jbf_morph_open": 3,
            "jbf_morph_close": 5,
            "jbf_blur_kernel": 3,
            "jbf_threshold": 230.0,
        }

    def filter_pipeline_names(self) -> list[str]:
        return [str(pipeline.get("name", "")) for pipeline in self.filter_pipelines]

    def selected_filter_pipeline(self) -> dict[str, object]:
        selected_name = self.main_filter_pipeline_var.get()
        for pipeline in self.filter_pipelines:
            if str(pipeline.get("name", "")) == selected_name:
                return deepcopy(pipeline)
        if self.filter_pipelines:
            fallback = deepcopy(self.filter_pipelines[0])
            self.main_filter_pipeline_var.set(str(fallback.get("name", "")))
            return fallback
        fallback = self.default_filter_pipeline()
        self.filter_pipelines.append(fallback)
        self.main_filter_pipeline_var.set(str(fallback["name"]))
        return deepcopy(fallback)

    def refresh_filter_pipeline_combo(self) -> None:
        names = self.filter_pipeline_names()
        if hasattr(self, "main_filter_combo"):
            self.main_filter_combo["values"] = names
        if self.main_filter_pipeline_var.get() not in names and names:
            self.main_filter_pipeline_var.set(names[0])
        self.clear_main_filter_state(clear_arrays=False)
        self.refresh_plot()

    @staticmethod
    def freeze_for_cache(value: object) -> object:
        if isinstance(value, dict):
            return tuple((key, GroupCameraFeatureExplorer.freeze_for_cache(value[key])) for key in sorted(value))
        if isinstance(value, list):
            return tuple(GroupCameraFeatureExplorer.freeze_for_cache(item) for item in value)
        if isinstance(value, float):
            return round(value, 6)
        return value

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

        main_filter_frame = ttk.LabelFrame(self.root, text="메인 Filter 배치 평가", padding=(8, 6))
        main_filter_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        main_filter_top = ttk.Frame(main_filter_frame)
        main_filter_top.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(
            main_filter_top,
            text="Filter 결과 표시",
            variable=self.main_filter_enable_var,
            command=self.refresh_plot,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(main_filter_top, text="Filter").pack(side=tk.LEFT, padx=(0, 4))
        self.main_filter_combo = ttk.Combobox(
            main_filter_top,
            textvariable=self.main_filter_pipeline_var,
            state="readonly",
            width=28,
            values=self.filter_pipeline_names(),
        )
        self.main_filter_combo.pack(side=tk.LEFT, padx=(0, 6))
        self.main_filter_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_plot())
        ttk.Button(main_filter_top, text="Filter 관리...", command=self.open_filter_manager).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(main_filter_top, text="Filter 적용/갱신", command=self.apply_main_filter_batch).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(main_filter_top, text="Sample rows(0=all)").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(main_filter_top, textvariable=self.main_filter_sample_size_var, width=8).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(main_filter_top, text="Seed").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(main_filter_top, textvariable=self.main_filter_seed_var, width=8).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(
            main_filter_top,
            text="Group 균등 샘플링",
            variable=self.main_filter_group_balanced_var,
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(
            main_filter_top,
            text="Filter Alive Checker",
            variable=self.main_filter_rate_line_var,
            command=self.refresh_plot,
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(main_filter_top, text="현재 Group/Camera 필터 대상에서 샘플링 후 Alive/Dead 평가").pack(side=tk.LEFT)

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
            "filter_alive",
            "filter_dead",
            "filter_eval",
            "filter_unknown",
            "filter_alive_rate",
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

    def open_filter_manager(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Filter Pipeline 관리")
        window.geometry("980x760")
        window.transient(self.root)

        outer = ttk.Frame(window, padding=10)
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(outer)
        list_frame.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        ttk.Label(list_frame, text="Pipelines").pack(side=tk.TOP, anchor="w")
        pipeline_list = tk.Listbox(list_frame, height=26, width=28, exportselection=False)
        pipeline_list.pack(side=tk.TOP, fill=tk.Y, expand=True)

        edit_frame = ttk.Frame(outer)
        edit_frame.grid(row=0, column=1, sticky="nsew")
        edit_frame.columnconfigure(1, weight=1)

        vars_map: dict[str, tk.Variable] = {
            "name": tk.StringVar(),
            "source": tk.StringVar(),
            "bbox_gate_enabled": tk.BooleanVar(),
            "bbox_gate_min_ratio": tk.DoubleVar(),
            "contrast": tk.DoubleVar(),
            "blur": tk.DoubleVar(),
            "kmeans_enabled": tk.BooleanVar(),
            "k": tk.IntVar(),
            "cluster_select": tk.StringVar(),
            "post_order": tk.StringVar(),
            "refine_enabled": tk.BooleanVar(),
            "refine_kernel": tk.IntVar(),
            "refine_kernel_w": tk.IntVar(),
            "refine_kernel_h": tk.IntVar(),
            "refine_angle_enabled": tk.BooleanVar(),
            "refine_open_iter": tk.IntVar(),
            "refine_close_iter": tk.IntVar(),
            "refine_min_area": tk.IntVar(),
            "jbf_enabled": tk.BooleanVar(),
            "jbf_diameter": tk.IntVar(),
            "jbf_sigma_color": tk.DoubleVar(),
            "jbf_sigma_space": tk.DoubleVar(),
            "jbf_morph_open": tk.IntVar(),
            "jbf_morph_close": tk.IntVar(),
            "jbf_blur_kernel": tk.IntVar(),
            "jbf_threshold": tk.DoubleVar(),
        }

        def set_vars(pipeline: dict[str, object]) -> None:
            defaults = self.default_filter_pipeline(str(pipeline.get("name", "Pipeline")))
            merged = {**defaults, **pipeline}
            for key, variable in vars_map.items():
                variable.set(merged.get(key, defaults.get(key, "")))

        def collect_pipeline() -> dict[str, object]:
            name = str(vars_map["name"].get()).strip() or "Unnamed Filter"
            return {
                "name": name,
                "source": str(vars_map["source"].get() or "image_path"),
                "bbox_gate_enabled": bool(vars_map["bbox_gate_enabled"].get()),
                "bbox_gate_min_ratio": float(vars_map["bbox_gate_min_ratio"].get()),
                "contrast": float(vars_map["contrast"].get()),
                "blur": float(vars_map["blur"].get()),
                "kmeans_enabled": bool(vars_map["kmeans_enabled"].get()),
                "k": int(vars_map["k"].get()),
                "cluster_select": str(vars_map["cluster_select"].get() or "all"),
                "post_order": str(vars_map["post_order"].get() or "jbf_then_refine"),
                "refine_enabled": bool(vars_map["refine_enabled"].get()),
                "refine_kernel": int(vars_map["refine_kernel"].get()),
                "refine_kernel_w": int(vars_map["refine_kernel_w"].get()),
                "refine_kernel_h": int(vars_map["refine_kernel_h"].get()),
                "refine_angle_enabled": bool(vars_map["refine_angle_enabled"].get()),
                "refine_open_iter": int(vars_map["refine_open_iter"].get()),
                "refine_close_iter": int(vars_map["refine_close_iter"].get()),
                "refine_min_area": int(vars_map["refine_min_area"].get()),
                "jbf_enabled": bool(vars_map["jbf_enabled"].get()),
                "jbf_diameter": int(vars_map["jbf_diameter"].get()),
                "jbf_sigma_color": float(vars_map["jbf_sigma_color"].get()),
                "jbf_sigma_space": float(vars_map["jbf_sigma_space"].get()),
                "jbf_morph_open": int(vars_map["jbf_morph_open"].get()),
                "jbf_morph_close": int(vars_map["jbf_morph_close"].get()),
                "jbf_blur_kernel": int(vars_map["jbf_blur_kernel"].get()),
                "jbf_threshold": float(vars_map["jbf_threshold"].get()),
            }

        def unique_name(base: str) -> str:
            existing = set(self.filter_pipeline_names())
            if base not in existing:
                return base
            idx = 2
            while f"{base} {idx}" in existing:
                idx += 1
            return f"{base} {idx}"

        def selected_index() -> int:
            selection = pipeline_list.curselection()
            return int(selection[0]) if selection else -1

        def refresh_list(select_idx: int | None = None) -> None:
            pipeline_list.delete(0, tk.END)
            for name in self.filter_pipeline_names():
                pipeline_list.insert(tk.END, name)
            if self.filter_pipelines:
                idx = 0 if select_idx is None else max(0, min(select_idx, len(self.filter_pipelines) - 1))
                pipeline_list.selection_clear(0, tk.END)
                pipeline_list.selection_set(idx)
                pipeline_list.activate(idx)
                set_vars(self.filter_pipelines[idx])

        def on_select(_event: object | None = None) -> None:
            idx = selected_index()
            if 0 <= idx < len(self.filter_pipelines):
                set_vars(self.filter_pipelines[idx])

        def add_pipeline() -> None:
            pipeline = self.default_filter_pipeline(unique_name("KMeans Filter"))
            pipeline["kmeans_enabled"] = True
            pipeline["k"] = 2
            pipeline["cluster_select"] = "darkest"
            pipeline["refine_enabled"] = True
            self.filter_pipelines.append(pipeline)
            refresh_list(len(self.filter_pipelines) - 1)

        def copy_pipeline() -> None:
            idx = selected_index()
            if idx < 0:
                return
            pipeline = deepcopy(self.filter_pipelines[idx])
            pipeline["name"] = unique_name(f"{pipeline.get('name', 'Pipeline')} Copy")
            self.filter_pipelines.append(pipeline)
            refresh_list(len(self.filter_pipelines) - 1)

        def save_pipeline() -> None:
            idx = selected_index()
            pipeline = collect_pipeline()
            names = self.filter_pipeline_names()
            if any(name == pipeline["name"] and pos != idx for pos, name in enumerate(names)):
                messagebox.showwarning("이름 중복", "같은 이름의 Filter Pipeline이 이미 있습니다.")
                return
            if idx < 0:
                self.filter_pipelines.append(pipeline)
                idx = len(self.filter_pipelines) - 1
            else:
                self.filter_pipelines[idx] = pipeline
            self.main_filter_pipeline_var.set(str(pipeline["name"]))
            self.refresh_filter_pipeline_combo()
            refresh_list(idx)

        def delete_pipeline() -> None:
            idx = selected_index()
            if idx < 0:
                return
            if len(self.filter_pipelines) <= 1:
                messagebox.showwarning("삭제 불가", "최소 1개의 Filter Pipeline은 필요합니다.")
                return
            del self.filter_pipelines[idx]
            self.refresh_filter_pipeline_combo()
            refresh_list(max(0, idx - 1))

        pipeline_list.bind("<<ListboxSelect>>", on_select)

        row = 0
        ttk.Label(edit_frame, text="Name").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(edit_frame, textvariable=vars_map["name"], width=34).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        ttk.Label(edit_frame, text="Source").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            edit_frame,
            textvariable=vars_map["source"],
            state="readonly",
            values=["image_path", "mask_raw_path"],
            width=20,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        bbox_gate = ttk.LabelFrame(edit_frame, text="BBox Ratio Gate", padding=(8, 6))
        bbox_gate.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        row += 1
        ttk.Checkbutton(
            bbox_gate,
            text="Use gate",
            variable=vars_map["bbox_gate_enabled"],
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(bbox_gate, text="Apply filter only if major/minor >=").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(
            bbox_gate,
            from_=1.0,
            to=50.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=vars_map["bbox_gate_min_ratio"],
            length=220,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(bbox_gate, text="Fail: keep original mask alive").pack(side=tk.LEFT)

        prep = ttk.LabelFrame(edit_frame, text="Preprocess", padding=(8, 6))
        prep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        row += 1
        ttk.Label(prep, text="Contrast").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(prep, from_=0.2, to=4.0, resolution=0.05, orient=tk.HORIZONTAL, variable=vars_map["contrast"], length=180).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(prep, text="Gaussian Blur").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(prep, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=vars_map["blur"], length=180).pack(side=tk.LEFT, padx=(0, 12))

        kmeans = ttk.LabelFrame(edit_frame, text="K-Means Layer", padding=(8, 6))
        kmeans.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Checkbutton(kmeans, text="Use K-Means", variable=vars_map["kmeans_enabled"]).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(kmeans, text="K").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(kmeans, from_=1, to=8, resolution=1, orient=tk.HORIZONTAL, variable=vars_map["k"], length=120).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(kmeans, text="Cluster").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            kmeans,
            textvariable=vars_map["cluster_select"],
            state="readonly",
            values=["all", "border_background", "darkest", "brightest", "largest", "smallest"],
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(kmeans, text="Order").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            kmeans,
            textvariable=vars_map["post_order"],
            state="readonly",
            values=["jbf_then_refine", "refine_then_jbf"],
            width=16,
        ).pack(side=tk.LEFT)

        refine = ttk.LabelFrame(edit_frame, text="Refinement Layer", padding=(8, 6))
        refine.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Checkbutton(refine, text="Use Refinement", variable=vars_map["refine_enabled"]).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(refine, text="Angle", variable=vars_map["refine_angle_enabled"]).pack(side=tk.LEFT, padx=(0, 10))
        for label, key, max_value in [
            ("Kernel", "refine_kernel", 15),
            ("W", "refine_kernel_w", 31),
            ("H", "refine_kernel_h", 31),
            ("Open", "refine_open_iter", 5),
            ("Close", "refine_close_iter", 5),
            ("MinArea", "refine_min_area", 500),
        ]:
            ttk.Label(refine, text=label).pack(side=tk.LEFT, padx=(0, 3))
            tk.Scale(refine, from_=0 if key in {"refine_open_iter", "refine_close_iter", "refine_min_area"} else 1, to=max_value, resolution=1, orient=tk.HORIZONTAL, variable=vars_map[key], length=80).pack(side=tk.LEFT, padx=(0, 6))

        jbf = ttk.LabelFrame(edit_frame, text="OpenCV JBF Layer", padding=(8, 6))
        jbf.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Checkbutton(jbf, text="Use JBF", variable=vars_map["jbf_enabled"]).pack(side=tk.LEFT, padx=(0, 10))
        for label, key, max_value, resolution in [
            ("D", "jbf_diameter", 31, 2),
            ("SC", "jbf_sigma_color", 150, 1),
            ("SS", "jbf_sigma_space", 50, 1),
            ("MO", "jbf_morph_open", 5, 1),
            ("MC", "jbf_morph_close", 5, 1),
            ("Blur", "jbf_blur_kernel", 31, 2),
            ("Th", "jbf_threshold", 255, 1),
        ]:
            ttk.Label(jbf, text=label).pack(side=tk.LEFT, padx=(0, 3))
            tk.Scale(jbf, from_=0 if key in {"jbf_morph_open", "jbf_morph_close", "jbf_threshold"} else 1, to=max_value, resolution=resolution, orient=tk.HORIZONTAL, variable=vars_map[key], length=80).pack(side=tk.LEFT, padx=(0, 6))

        button_row = ttk.Frame(outer)
        button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_row, text="새 Pipeline", command=add_pipeline).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row, text="복사", command=copy_pipeline).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row, text="저장", command=save_pipeline).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row, text="삭제", command=delete_pipeline).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Button(button_row, text="닫기", command=window.destroy).pack(side=tk.RIGHT)

        refresh_list(0)

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
            self.clear_main_filter_state(clear_arrays=True)
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
        self.image_search_cache.clear()
        self.clear_main_filter_state(clear_arrays=True)
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

    @staticmethod
    def parse_int_text(raw: str, default: int = 0, minimum: int = 0) -> int:
        try:
            value = int(str(raw).strip())
        except ValueError:
            return default
        return max(value, minimum)

    def main_filter_params_tuple(self, pipeline: dict[str, object]) -> tuple[object, ...]:
        return (MAIN_FILTER_CACHE_VERSION, self.freeze_for_cache(pipeline))

    @staticmethod
    def bbox_ratio_from_mask(mask: np.ndarray) -> tuple[float, int, int]:
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return 0.0, 0, 0
        coords_yx: np.ndarray
        if cv2 is not None:
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask_bool.astype(np.uint8),
                connectivity=8,
            )
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                label_id = int(np.argmax(areas)) + 1
                coords_yx = np.column_stack(np.where(labels == label_id)).astype(np.int32)
            else:
                coords_yx = np.column_stack(np.where(mask_bool)).astype(np.int32)
        else:
            components = ImageViewer.connected_components_bool(mask_bool, min_area=8)
            coords_yx = max(components, key=len) if components else np.column_stack(np.where(mask_bool)).astype(np.int32)

        geometry = ImageViewer.component_angle_geometry(coords_yx)
        major = max(float(geometry["major_length"]), 1.0)
        minor = max(float(geometry["minor_length"]), 1.0)
        return float(major / minor), int(round(major)), int(round(minor))

    def bbox_gate_decision(self, mask: np.ndarray, pipeline: dict[str, object]) -> tuple[bool, str, float]:
        ratio, width, height = self.bbox_ratio_from_mask(mask)
        if not bool(pipeline.get("bbox_gate_enabled", False)):
            return True, f"bbox_gate=off(oriented R={ratio:.2f},{width}x{height})", ratio
        threshold = max(float(pipeline.get("bbox_gate_min_ratio", 10.0)), 1.0)
        passed = ratio >= threshold
        action = "pass" if passed else "skip_keep"
        comparator = ">=" if passed else "<"
        return passed, f"bbox_gate={action}(oriented R={ratio:.2f}{comparator}{threshold:.2f},{width}x{height})", ratio

    def clear_main_filter_state(self, clear_arrays: bool = False) -> None:
        self.main_filter_results.clear()
        self.main_filter_cache.clear()
        self.main_filter_scope_text = ""
        if clear_arrays:
            self.main_filter_image_cache.clear()
            self.main_filter_mask_cache.clear()

    @staticmethod
    def lru_get(cache: OrderedDict[tuple[object, ...], object], key: tuple[object, ...]) -> tuple[bool, object]:
        if key not in cache:
            return False, None
        value = cache.pop(key)
        cache[key] = value
        return True, value

    @staticmethod
    def lru_put(
        cache: OrderedDict[tuple[object, ...], object],
        key: tuple[object, ...],
        value: object,
        limit: int,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def main_filter_file_token(path: Path | None) -> tuple[object, ...]:
        if path is None:
            return ("missing",)
        try:
            resolved = str(path.resolve())
        except Exception:  # noqa: BLE001 - path normalization should not block batch evaluation
            resolved = str(path)
        try:
            stat = path.stat()
        except OSError:
            return ("not_found", resolved)
        return ("file", resolved, int(stat.st_size), int(stat.st_mtime_ns))

    def cached_main_filter_rgb_array(
        self,
        path: Path | None,
        expected_shape: tuple[int, int] | None,
        file_token: tuple[object, ...],
    ) -> np.ndarray | None:
        shape_key = tuple(expected_shape) if expected_shape is not None else None
        key = ("rgb", file_token, shape_key)
        hit, value = self.lru_get(self.main_filter_image_cache, key)
        if hit:
            return value  # type: ignore[return-value]
        value = ImageViewer.load_rgb_array(path, expected_shape)
        self.lru_put(self.main_filter_image_cache, key, value, self.main_filter_image_cache_limit)
        return value

    def cached_main_filter_mask_array(
        self,
        path: Path | None,
        expected_shape: tuple[int, int],
        file_token: tuple[object, ...],
    ) -> np.ndarray | None:
        key = ("mask", file_token, tuple(expected_shape))
        hit, value = self.lru_get(self.main_filter_mask_cache, key)
        if hit:
            return value  # type: ignore[return-value]
        value = ImageViewer.load_mask_array(path, expected_shape)
        self.lru_put(self.main_filter_mask_cache, key, value, self.main_filter_mask_cache_limit)
        return value

    def get_main_filter_result_cache(self, key: tuple[object, ...]) -> dict[str, object] | None:
        hit, value = self.lru_get(self.main_filter_cache, key)
        if not hit:
            return None
        cached = dict(value)  # type: ignore[arg-type]
        cached["cache_hit"] = True
        return cached

    def set_main_filter_result_cache(self, key: tuple[object, ...], result: dict[str, object]) -> None:
        stored = dict(result)
        stored["cache_hit"] = False
        self.lru_put(self.main_filter_cache, key, stored, self.main_filter_result_cache_limit)

    @staticmethod
    def kmeans_labels_for_filter(
        array: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        k = int(np.clip(k, 1, 8))
        h, w, _ = array.shape
        labels_2d = np.full((h, w), -1, dtype=np.int16)
        pixels = array[valid_mask].reshape(-1, 3).astype(np.float32)
        if len(pixels) <= 0:
            return labels_2d, np.zeros((k, 3), dtype=np.float32), np.zeros(k, dtype=np.int64), 0.0

        if k <= 1:
            ordered_labels = np.zeros(len(pixels), dtype=np.int16)
            ordered_centers = pixels.mean(axis=0, keepdims=True).astype(np.float32)
        else:
            rng = np.random.default_rng(17)
            sample_size = min(25000, len(pixels))
            sample_idx = rng.choice(len(pixels), size=sample_size, replace=False)
            sample = pixels[sample_idx]
            luma = sample @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            centers = []
            for q in np.linspace(0.05, 0.95, k):
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
            for chunk_start in range(0, len(pixels), chunk):
                chunk_stop = min(chunk_start + chunk, len(pixels))
                distances = ((pixels[chunk_start:chunk_stop, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                all_labels[chunk_start:chunk_stop] = distances.argmin(axis=1)

            center_luma = centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            order = np.argsort(center_luma)
            remap = np.zeros(k, dtype=np.int16)
            for new_id, old_id in enumerate(order):
                remap[old_id] = new_id
            ordered_labels = remap[all_labels]
            ordered_centers = centers[order]

        labels_2d[valid_mask] = ordered_labels
        counts = np.bincount(ordered_labels, minlength=len(ordered_centers))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return labels_2d, ordered_centers, counts, elapsed_ms

    @staticmethod
    def select_kmeans_candidate(
        labels_2d: np.ndarray,
        valid_mask: np.ndarray,
        counts: np.ndarray,
        cluster_select: str,
        centers: np.ndarray | None = None,
        image_array: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str]:
        if counts.size <= 0:
            return np.zeros_like(valid_mask, dtype=bool), "cluster=none"
        select = str(cluster_select or "all")
        nonzero = np.where(counts > 0)[0]
        if select == "border_background" and centers is not None and image_array is not None:
            bg_cluster_id, bg_text = ImageViewer.infer_border_background_cluster(
                labels_2d,
                valid_mask,
                centers,
                image_array,
                counts,
            )
            if bg_cluster_id is not None:
                candidate = valid_mask & (labels_2d != int(bg_cluster_id))
                return candidate, bg_text
            select = "largest"

        if select == "darkest":
            selected = [0]
        elif select == "brightest":
            selected = [int(len(counts) - 1)]
        elif select == "largest":
            selected = [int(np.argmax(counts))]
        elif select == "smallest":
            selected = [int(nonzero[np.argmin(counts[nonzero])])] if len(nonzero) else [0]
        else:
            selected = list(range(len(counts)))
            select = "all"
        candidate = np.isin(labels_2d, selected) & valid_mask
        return candidate, f"cluster={select}:{','.join(str(idx) for idx in selected)}"

    @staticmethod
    def apply_filter_refinement(mask: np.ndarray, angle_mask: np.ndarray, pipeline: dict[str, object]) -> tuple[np.ndarray, str]:
        if not bool(pipeline.get("refine_enabled", False)):
            return mask.astype(bool), "refine=off"
        if bool(pipeline.get("refine_angle_enabled", False)):
            clean, component_count = ImageViewer.morphology_clean_angle_aligned(
                mask,
                angle_mask=angle_mask,
                kernel_size=int(pipeline.get("refine_kernel", 3)),
                kernel_w=int(pipeline.get("refine_kernel_w", 3)),
                kernel_h=int(pipeline.get("refine_kernel_h", 3)),
                open_iter=int(pipeline.get("refine_open_iter", 1)),
                close_iter=int(pipeline.get("refine_close_iter", 1)),
            )
            mode = f"refine=angle({component_count})"
        else:
            clean = ImageViewer.morphology_clean(
                mask,
                kernel_size=int(pipeline.get("refine_kernel", 3)),
                kernel_w=int(pipeline.get("refine_kernel_w", 3)),
                kernel_h=int(pipeline.get("refine_kernel_h", 3)),
                open_iter=int(pipeline.get("refine_open_iter", 1)),
                close_iter=int(pipeline.get("refine_close_iter", 1)),
            )
            mode = "refine=axis"
        clean = ImageViewer.remove_small_components(clean, min_area=int(pipeline.get("refine_min_area", 12)))
        return clean, mode

    @staticmethod
    def apply_filter_jbf(
        mask: np.ndarray,
        guide: np.ndarray,
        valid_mask: np.ndarray,
        pipeline: dict[str, object],
    ) -> tuple[np.ndarray, str, bool]:
        if not bool(pipeline.get("jbf_enabled", False)):
            return mask.astype(bool), "jbf=off", False
        refined_mask, _elapsed_ms, mode = ImageViewer.jbf_refine_mask(
            mask,
            guide_array=guide,
            valid_mask=valid_mask,
            diameter=int(pipeline.get("jbf_diameter", 15)),
            sigma_color=float(pipeline.get("jbf_sigma_color", 30.0)),
            sigma_space=float(pipeline.get("jbf_sigma_space", 15.0)),
            morph_open=int(pipeline.get("jbf_morph_open", 3)),
            morph_close=int(pipeline.get("jbf_morph_close", 5)),
            blur_kernel=int(pipeline.get("jbf_blur_kernel", 3)),
            threshold=float(pipeline.get("jbf_threshold", 230.0)),
            require_opencv_jbf=True,
        )
        failed = mode in {"opencv-jbf-unavailable", "opencv-jbf-error"}
        return refined_mask, f"jbf={mode}", failed

    def render_filter_pipeline_preview(
        self,
        source_array: np.ndarray,
        mask_array: np.ndarray | None,
        pipeline: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        start = time.perf_counter()
        adjusted = ImageViewer.apply_blur(
            ImageViewer.apply_contrast(source_array, float(pipeline.get("contrast", 1.0))),
            float(pipeline.get("blur", 0.0)),
        )
        h, w, _ = adjusted.shape
        if mask_array is not None and mask_array.shape == (h, w) and mask_array.any():
            valid_mask = mask_array.astype(bool)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        k = int(pipeline.get("k", 1)) if bool(pipeline.get("kmeans_enabled", False)) else 1
        mode_parts: list[str] = []
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
        gate_pass, gate_text, _bbox_ratio = self.bbox_gate_decision(valid_mask, pipeline)
        mode_parts.append(gate_text)
        if not gate_pass:
            raw_area = int(valid_mask.sum())
            clustered[valid_mask] = adjusted[valid_mask]
            refined = np.zeros((h, w, 3), dtype=np.uint8)
            refined[valid_mask] = np.array([44, 160, 44], dtype=np.uint8)
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats_text = (
                f"order=skipped_by_bbox_gate, raw_area={raw_area}, alive_area={raw_area}, "
                f"status=alive, elapsed={elapsed_ms:.1f}ms | {' | '.join(mode_parts)}"
            )
            return adjusted, clustered, refined, stats_text

        if k > 1:
            labels_2d, centers, counts, kmeans_ms = self.kmeans_labels_for_filter(adjusted, valid_mask, k)
            for cluster_id in range(min(len(counts), len(palette))):
                clustered[(labels_2d == cluster_id) & valid_mask] = palette[cluster_id]
            candidate, cluster_text = self.select_kmeans_candidate(
                labels_2d,
                valid_mask,
                counts,
                str(pipeline.get("cluster_select", "all")),
                centers=centers,
                image_array=adjusted,
            )
            if str(pipeline.get("cluster_select", "all")) == "border_background":
                clustered[valid_mask & ~candidate] = 0
            if str(pipeline.get("cluster_select", "all")) != "all":
                selected_overlay = candidate & valid_mask
                clustered[selected_overlay] = np.clip(
                    clustered[selected_overlay].astype(np.float32) * 0.45 + np.array([255, 255, 255]) * 0.55,
                    0,
                    255,
                ).astype(np.uint8)
            mode_parts.append(f"kmeans={kmeans_ms:.1f}ms")
            mode_parts.append(cluster_text)
        else:
            candidate = valid_mask.copy()
            counts = np.asarray([int(valid_mask.sum())], dtype=np.int64)
            clustered[valid_mask] = adjusted[valid_mask]
            mode_parts.append("kmeans=off")
            mode_parts.append("cluster=all")

        raw_area = int(candidate.sum())
        work_mask = candidate
        failed = False
        order = str(pipeline.get("post_order", "jbf_then_refine"))
        if order == "refine_then_jbf":
            work_mask, refine_mode = self.apply_filter_refinement(work_mask, valid_mask, pipeline)
            mode_parts.append(refine_mode)
            work_mask, jbf_mode, failed = self.apply_filter_jbf(work_mask, adjusted, valid_mask, pipeline)
            mode_parts.append(jbf_mode)
        else:
            work_mask, jbf_mode, failed = self.apply_filter_jbf(work_mask, adjusted, valid_mask, pipeline)
            mode_parts.append(jbf_mode)
            work_mask, refine_mode = self.apply_filter_refinement(work_mask, valid_mask, pipeline)
            mode_parts.append(refine_mode)

        refined = np.zeros((h, w, 3), dtype=np.uint8)
        if failed:
            refined[candidate] = np.array([214, 39, 40], dtype=np.uint8)
        else:
            refined[work_mask] = np.array([44, 160, 44], dtype=np.uint8)
        elapsed_ms = (time.perf_counter() - start) * 1000
        alive_area = int(work_mask.sum()) if not failed else 0
        stats_text = (
            f"order={order}, raw_area={raw_area}, alive_area={alive_area}, "
            f"status={'unknown' if failed else ('alive' if alive_area > 0 else 'dead')}, "
            f"elapsed={elapsed_ms:.1f}ms | {' | '.join(mode_parts)}"
        )
        return adjusted, clustered, refined, stats_text

    def evaluate_main_filter_row(self, row: pd.Series, pipeline: dict[str, object], params_key: tuple[object, ...]) -> dict[str, object]:
        row_id = int(row.get("row_id"))
        source_name = str(pipeline.get("source", "image_path"))
        source_path = self.resolve_image_path(row.get(source_name)) if source_name in row.index else None
        if source_path is None and "image_path" in row.index:
            source_path = self.resolve_image_path(row.get("image_path"))
        if source_path is None and "mask_raw_path" in row.index:
            source_path = self.resolve_image_path(row.get("mask_raw_path"))
        mask_path = self.resolve_image_path(row.get("mask_path")) if "mask_path" in row.index else None
        source_token = self.main_filter_file_token(source_path)
        mask_token = self.main_filter_file_token(mask_path)
        cache_key = (MAIN_FILTER_CACHE_VERSION, row_id, source_token, mask_token, params_key)
        cached_result = self.get_main_filter_result_cache(cache_key)
        if cached_result is not None:
            return cached_result

        filter_name = str(pipeline.get("name", "Filter"))
        if source_path is None or mask_path is None:
            result = {
                "filter_name": filter_name,
                "status": "unknown",
                "raw_area": 0,
                "alive_area": 0,
                "elapsed_ms": 0.0,
                "mode": "missing_path",
                "cache_hit": False,
            }
            self.set_main_filter_result_cache(cache_key, result)
            return result

        start = time.perf_counter()
        try:
            guide = self.cached_main_filter_rgb_array(source_path, None, source_token)
            if guide is None:
                raise FileNotFoundError(str(source_path))
            mask = self.cached_main_filter_mask_array(mask_path, guide.shape[:2], mask_token)
            if mask is None:
                raise FileNotFoundError(str(mask_path))
            mask = mask.astype(bool)
            mask_area = int(mask.sum())
            if mask_area <= 0:
                result = {
                    "filter_name": filter_name,
                    "status": "dead",
                    "raw_area": 0,
                    "alive_area": 0,
                    "elapsed_ms": 0.0,
                    "mode": "empty_mask",
                    "cache_hit": False,
                }
            else:
                mode_parts = [f"source={source_name}"]
                gate_pass, gate_text, _bbox_ratio = self.bbox_gate_decision(mask, pipeline)
                mode_parts.append(gate_text)
                if not gate_pass:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    result = {
                        "filter_name": filter_name,
                        "status": "alive",
                        "raw_area": mask_area,
                        "alive_area": mask_area,
                        "elapsed_ms": float(elapsed_ms),
                        "mode": " | ".join(mode_parts),
                        "cache_hit": False,
                    }
                    self.set_main_filter_result_cache(cache_key, result)
                    return result

                adjusted = ImageViewer.apply_blur(
                    ImageViewer.apply_contrast(guide, float(pipeline.get("contrast", 1.0))),
                    float(pipeline.get("blur", 0.0)),
                )
                if bool(pipeline.get("kmeans_enabled", False)) and int(pipeline.get("k", 1)) > 1:
                    labels_2d, centers, counts, kmeans_ms = self.kmeans_labels_for_filter(adjusted, mask, int(pipeline.get("k", 2)))
                    candidate, cluster_text = self.select_kmeans_candidate(
                        labels_2d,
                        mask,
                        counts,
                        str(pipeline.get("cluster_select", "all")),
                        centers=centers,
                        image_array=adjusted,
                    )
                    mode_parts.append(f"kmeans={kmeans_ms:.1f}ms")
                    mode_parts.append(cluster_text)
                else:
                    candidate = mask.copy()
                    mode_parts.append("kmeans=off")

                raw_area = int(candidate.sum())
                work_mask = candidate
                failed = False
                order = str(pipeline.get("post_order", "jbf_then_refine"))
                if order == "refine_then_jbf":
                    work_mask, refine_mode = self.apply_filter_refinement(work_mask, mask, pipeline)
                    mode_parts.append(refine_mode)
                    work_mask, jbf_mode, failed = self.apply_filter_jbf(work_mask, adjusted, mask, pipeline)
                    mode_parts.append(jbf_mode)
                else:
                    work_mask, jbf_mode, failed = self.apply_filter_jbf(work_mask, adjusted, mask, pipeline)
                    mode_parts.append(jbf_mode)
                    work_mask, refine_mode = self.apply_filter_refinement(work_mask, mask, pipeline)
                    mode_parts.append(refine_mode)

                elapsed_ms = (time.perf_counter() - start) * 1000
                alive_area = int(work_mask.sum())
                result = {
                    "filter_name": filter_name,
                    "status": "unknown" if failed else ("alive" if alive_area > 0 else "dead"),
                    "raw_area": raw_area,
                    "alive_area": 0 if failed else alive_area,
                    "elapsed_ms": float(elapsed_ms),
                    "mode": " | ".join(mode_parts),
                    "cache_hit": False,
                }
        except Exception as exc:  # noqa: BLE001 - batch evaluation should continue per row
            result = {
                "filter_name": filter_name,
                "status": "unknown",
                "raw_area": 0,
                "alive_area": 0,
                "elapsed_ms": 0.0,
                "mode": f"error:{type(exc).__name__}",
                "cache_hit": False,
            }

        self.set_main_filter_result_cache(cache_key, result)
        return result

    @staticmethod
    def group_balanced_sample(work: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
        if sample_size <= 0 or len(work) <= sample_size or "group" not in work.columns:
            return work.copy()

        groups = [(group_key, part.copy()) for group_key, part in work.groupby("group", sort=False, dropna=False)]
        if not groups:
            return work.head(0).copy()

        rng = np.random.default_rng(seed)
        capacities = np.asarray([len(part) for _group_key, part in groups], dtype=np.int64)
        allocation = np.zeros(len(groups), dtype=np.int64)
        remaining = min(int(sample_size), int(capacities.sum()))
        order = np.arange(len(groups))
        rng.shuffle(order)

        while remaining > 0:
            active = [int(idx) for idx in order if allocation[int(idx)] < capacities[int(idx)]]
            if not active:
                break
            take_each = max(1, remaining // len(active))
            progressed = False
            for idx in active:
                available = int(capacities[idx] - allocation[idx])
                take = min(take_each, available, remaining)
                if take <= 0:
                    continue
                allocation[idx] += take
                remaining -= take
                progressed = True
                if remaining <= 0:
                    break
            if not progressed:
                break

        sampled_parts = []
        for idx, (_group_key, part) in enumerate(groups):
            n = int(allocation[idx])
            if n <= 0:
                continue
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            sampled_parts.append(part.sample(n=n, random_state=random_state))

        if not sampled_parts:
            return work.head(0).copy()
        sampled = pd.concat(sampled_parts, axis=0)
        return sampled.sample(frac=1.0, random_state=seed)

    def apply_main_filter_batch(self) -> None:
        if self.df.empty:
            messagebox.showwarning("CSV 필요", "먼저 CSV를 로드하세요.")
            return
        if "mask_path" not in self.df.columns:
            messagebox.showwarning("mask_path 필요", "CSV에 mask_path 컬럼이 필요합니다.")
            return
        pipeline = self.selected_filter_pipeline()
        if bool(pipeline.get("jbf_enabled", False)) and not ImageViewer.has_opencv_joint_bilateral_filter():
            install_text = f"{OPENCV_JBF_INSTALL_HINT}\n\n설치 예: python -m pip install opencv-contrib-python"
            messagebox.showwarning("OpenCV JBF 필요", install_text)
            self.status_var.set("Filter 평가 중단: opencv-contrib-python/cv2.ximgproc가 필요합니다.")
            return

        work = self.filtered().copy()
        if work.empty:
            self.status_var.set("Filter 평가 대상 row가 없습니다.")
            return

        candidate_count = len(work)
        sample_size = self.parse_int_text(self.main_filter_sample_size_var.get(), default=500, minimum=0)
        seed = self.parse_int_text(self.main_filter_seed_var.get(), default=17, minimum=0)
        if sample_size > 0 and len(work) > sample_size:
            if self.main_filter_group_balanced_var.get():
                work = self.group_balanced_sample(work, sample_size=sample_size, seed=seed)
                sampled_text = f"sampled={len(work)}/{candidate_count}, seed={seed}, group-balanced"
            else:
                work = work.sample(n=sample_size, random_state=seed)
                sampled_text = f"sampled={sample_size}/{candidate_count}, seed={seed}, random"
        else:
            mode_text = "group-balanced" if self.main_filter_group_balanced_var.get() else "random"
            sampled_text = f"sampled=all({len(work)}), seed={seed}, {mode_text}"

        params_key = self.main_filter_params_tuple(pipeline)
        filter_name = str(pipeline.get("name", "Filter"))
        self.main_filter_results.clear()
        self.main_filter_scope_text = f"{sampled_text}, filter={filter_name}"
        start = time.perf_counter()
        total = len(work)
        cache_hits = 0
        for index, (_idx, row) in enumerate(work.iterrows(), start=1):
            row_id = int(row.get("row_id"))
            result = self.evaluate_main_filter_row(row, pipeline, params_key)
            cache_hits += int(bool(result.get("cache_hit", False)))
            self.main_filter_results[row_id] = result
            if index == 1 or index % 10 == 0 or index == total:
                self.status_var.set(f"Filter 평가 중 {index}/{total} | {sampled_text} | {filter_name}")
                self.root.update_idletasks()

        elapsed_ms = (time.perf_counter() - start) * 1000
        statuses = pd.Series([str(result.get("status", "unknown")) for result in self.main_filter_results.values()])
        alive = int((statuses == "alive").sum())
        dead = int((statuses == "dead").sum())
        unknown = int((statuses == "unknown").sum())
        self.main_filter_enable_var.set(True)
        self.refresh_plot()
        self.status_var.set(
            f"Filter 평가 완료 | {sampled_text} | filter={filter_name} | alive={alive}, dead={dead}, unknown={unknown}, "
            f"cache_hit={cache_hits}/{total}, elapsed={elapsed_ms:.1f}ms"
        )

    def attach_main_filter_results(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        if plot_df.empty or not self.main_filter_enable_var.get():
            return plot_df
        if self.main_filter_results:
            result_ids = set(self.main_filter_results)
            plot_df = plot_df[plot_df["row_id"].astype(int).isin(result_ids)].copy()
            if plot_df.empty:
                return plot_df
        out = plot_df.copy()
        statuses = []
        raw_areas = []
        alive_areas = []
        modes = []
        filter_names = []
        for row_id in out["row_id"].astype(int):
            result = self.main_filter_results.get(int(row_id))
            if result is None:
                statuses.append("unknown")
                raw_areas.append(np.nan)
                alive_areas.append(np.nan)
                modes.append("")
                filter_names.append("")
            else:
                statuses.append(str(result.get("status", "unknown")))
                raw_areas.append(result.get("raw_area", np.nan))
                alive_areas.append(result.get("alive_area", np.nan))
                modes.append(str(result.get("mode", "")))
                filter_names.append(str(result.get("filter_name", "")))
        out["filter_status"] = statuses
        out["filter_raw_area"] = raw_areas
        out["filter_alive_area"] = alive_areas
        out["filter_mode"] = modes
        out["filter_name"] = filter_names
        return out

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
        plot_df = self.attach_main_filter_results(plot_df)
        threshold = self.parse_threshold()

        if self.filter_rate_ax is not None:
            try:
                self.filter_rate_ax.remove()
            except ValueError:
                pass
            self.filter_rate_ax = None
        self.ax.clear()
        self.ax.patch.set_visible(True)
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

        if self.main_filter_rate_line_var.get() and "filter_status" in plot_df.columns and not summary.empty:
            rate_rows = []
            for _, row in summary.iterrows():
                cat = (str(row["group"]), str(row["camera_mode"]))
                x = cat_to_x.get(cat)
                eval_count = safe_float(row.get("filter_eval", np.nan))
                alive_rate = safe_float(row.get("filter_alive_rate", np.nan))
                if x is None or not np.isfinite(eval_count) or eval_count <= 0 or not np.isfinite(alive_rate):
                    continue
                rate_rows.append((x, alive_rate * 100.0, (1.0 - alive_rate) * 100.0))
            if rate_rows:
                rate_rows.sort(key=lambda item: item[0])
                xs = [item[0] for item in rate_rows]
                alive_pct = [item[1] for item in rate_rows]
                dead_pct = [item[2] for item in rate_rows]
                self.filter_rate_ax = self.ax.twinx()
                self.filter_rate_ax.set_zorder(0)
                self.filter_rate_ax.patch.set_visible(False)
                self.ax.set_zorder(1)
                self.ax.patch.set_visible(False)
                self.filter_rate_ax.plot(
                    xs,
                    alive_pct,
                    color="#2ca02c",
                    linestyle="-",
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    alpha=0.82,
                    label="Filter Alive %",
                    zorder=0,
                )
                self.filter_rate_ax.plot(
                    xs,
                    dead_pct,
                    color="#d62728",
                    linestyle="-",
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    alpha=0.82,
                    label="Filter Dead %",
                    zorder=0,
                )
                rate_values = alive_pct + dead_pct
                rate_min = min(rate_values)
                rate_max = max(rate_values)
                rate_margin = max((rate_max - rate_min) * 0.12, 5.0)
                y_min = max(0.0, rate_min - rate_margin)
                y_max = min(100.0, rate_max + rate_margin)
                if y_max - y_min < 20.0:
                    center = (y_min + y_max) / 2.0
                    y_min = max(0.0, center - 10.0)
                    y_max = min(100.0, center + 10.0)
                if rate_max >= 98.0:
                    y_min = min(y_min, 90.0)
                if rate_min <= 2.0:
                    y_max = max(y_max, 10.0)
                self.filter_rate_ax.set_ylim(y_min, y_max)
                self.filter_rate_ax.set_ylabel("Filter Alive/Dead (%)")
                self.filter_rate_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                self.filter_rate_ax.grid(False)

        rng = np.random.default_rng(7)
        plot_df["x"] = plot_df["x_base"].astype(float) + rng.uniform(-0.22, 0.22, size=len(plot_df))

        for defect_type, part in plot_df.groupby("defect_type", sort=False):
            color = DEFECT_COLORS.get(defect_type, "#666666")
            if self.main_filter_enable_var.get() and "filter_status" in part.columns:
                edgecolors = part["filter_status"].map({"alive": "#2ca02c", "dead": "#d62728", "unknown": "#9e9e9e"}).fillna("#9e9e9e").tolist()
                linewidths = part["filter_status"].map({"alive": 1.35, "dead": 1.35, "unknown": 0.5}).fillna(0.5).tolist()
            else:
                edgecolors = "white"
                linewidths = 0.45
            artist = self.ax.scatter(
                part["x"],
                part[feature],
                s=34,
                alpha=0.72,
                label=f"{defect_type} ({len(part)})",
                color=color,
                edgecolors=edgecolors,
                linewidths=linewidths,
                zorder=4,
                picker=True,
                pickradius=6,
            )
            self.artist_rows[artist] = part["row_id"].astype(int).tolist()

        filter_status = ""
        if self.main_filter_enable_var.get() and "filter_status" in plot_df.columns:
            alive = int((plot_df["filter_status"] == "alive").sum())
            dead = int((plot_df["filter_status"] == "dead").sum())
            unknown = int((plot_df["filter_status"] == "unknown").sum())
            evaluated = alive + dead
            scope_text = f", {self.main_filter_scope_text}" if self.main_filter_scope_text else ""
            filter_status = f" | Filter eval={evaluated}, alive={alive}, dead={dead}, unknown={unknown}{scope_text}"
            self.ax.scatter([], [], s=54, facecolors="none", edgecolors="#2ca02c", linewidths=1.5, label=f"Filter Alive ({alive})")
            self.ax.scatter([], [], s=54, facecolors="none", edgecolors="#d62728", linewidths=1.5, label=f"Filter Dead ({dead})")

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
        handles, legend_labels = self.ax.get_legend_handles_labels()
        if self.filter_rate_ax is not None:
            rate_handles, rate_labels = self.filter_rate_ax.get_legend_handles_labels()
            handles += rate_handles
            legend_labels += rate_labels
        self.ax.legend(handles, legend_labels, loc="best", fontsize=8)
        self.ax.grid(True, alpha=0.28)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(
            f"표시 rows={len(plot_df)} | groups={plot_df['group'].nunique()} | cameras={plot_df['camera_mode'].nunique()} | 점 클릭 시 이미지 창 갱신{threshold_status}{filter_status}"
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
        for col in ["filter_eval", "filter_alive", "filter_dead", "filter_unknown", "filter_alive_rate"]:
            summary[col] = np.nan
        if "filter_status" in plot_df.columns:
            filter_counts = (
                plot_df.pivot_table(
                    index=["group", "camera_mode"],
                    columns="filter_status",
                    values="row_id",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            summary = summary.drop(columns=["filter_eval", "filter_alive", "filter_dead", "filter_unknown", "filter_alive_rate"], errors="ignore")
            summary = summary.merge(filter_counts, on=["group", "camera_mode"], how="left")
            for status in ["alive", "dead", "unknown"]:
                if status not in summary.columns:
                    summary[status] = 0
            summary["filter_alive"] = pd.to_numeric(summary["alive"], errors="coerce").fillna(0).astype(int)
            summary["filter_dead"] = pd.to_numeric(summary["dead"], errors="coerce").fillna(0).astype(int)
            summary["filter_unknown"] = pd.to_numeric(summary["unknown"], errors="coerce").fillna(0).astype(int)
            summary["filter_eval"] = summary["filter_alive"] + summary["filter_dead"]
            summary["filter_alive_rate"] = summary["filter_alive"] / summary["filter_eval"].replace(0, np.nan)
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
                "" if pd.isna(row.get("filter_alive", np.nan)) else int(row.get("filter_alive", 0)),
                "" if pd.isna(row.get("filter_dead", np.nan)) else int(row.get("filter_dead", 0)),
                "" if pd.isna(row.get("filter_eval", np.nan)) else int(row.get("filter_eval", 0)),
                "" if pd.isna(row.get("filter_unknown", np.nan)) else int(row.get("filter_unknown", 0)),
                "" if pd.isna(row.get("filter_alive_rate", np.nan)) else self.format_percent(row.get("filter_alive_rate")),
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
