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
    from PIL import Image, ImageTk
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
        self.image_label: ttk.Label | None = None
        self.meta_label: ttk.Label | None = None
        self.photo: ImageTk.PhotoImage | None = None

    def _ensure_window(self) -> None:
        if self.window is not None and self.window.winfo_exists():
            return

        self.window = tk.Toplevel(self.master)
        self.window.title("선택 이미지")
        self.window.geometry("980x820")
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        self.meta_label = ttk.Label(self.window, text="", anchor="w", justify="left")
        self.meta_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.image_label = ttk.Label(self.window, anchor="center")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

    def show_error(self, message: str) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None and self.image_label is not None
        self.photo = None
        self.meta_label.config(text=message)
        self.image_label.config(image="", text="이미지를 불러오지 못했습니다.")
        self.window.deiconify()
        self.window.lift()

    def update(self, image_path: Path, meta_text: str) -> None:
        self._ensure_window()
        assert self.window is not None and self.meta_label is not None and self.image_label is not None

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((940, 720), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"{meta_text}\n\n이미지 로딩 실패: {image_path}\n{exc}")
            return

        self.meta_label.config(text=f"{meta_text}\nimage: {image_path}")
        self.image_label.config(image=self.photo, text="")
        self.window.deiconify()
        self.window.lift()


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
        self.artist_rows: dict[object, list[int]] = {}
        self.image_viewer = ImageViewer(root)

        self.csv_path_var = tk.StringVar(value=str(csv_path) if csv_path else "")
        self.image_root_var = tk.StringVar(value=str(image_root) if image_root else "")
        self.feature_var = tk.StringVar()
        self.group_filter_var = tk.StringVar(value="전체")
        self.camera_filter_var = tk.StringVar(value="전체")
        self.sort_var = tk.StringVar(value="group_camera")
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

        if not self.numeric_features:
            messagebox.showerror("수치형 feature 없음", "그래프로 표시할 numeric feature가 없습니다.")
            return

        self.feature_combo["values"] = self.numeric_features
        default_feature = self._default_feature(self.numeric_features)
        self.feature_var.set(default_feature)

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

        self.ax.clear()
        self.artist_rows.clear()
        self.update_summary_table(plot_df, feature)

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
            f"표시 rows={len(plot_df)} | groups={plot_df['group'].nunique()} | cameras={plot_df['camera_mode'].nunique()} | 점 클릭 시 이미지 창 갱신"
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

    def on_pick(self, event) -> None:  # noqa: ANN001 - matplotlib event
        artist = event.artist
        if artist not in self.artist_rows or not len(event.ind):
            return
        local_idx = int(event.ind[0])
        row_id = self.artist_rows[artist][local_idx]
        row = self.df.loc[self.df["row_id"] == row_id].iloc[0]
        self.show_row_image(row)

    def show_row_image(self, row: pd.Series) -> None:
        image_path = self.resolve_image_path(row.get("image_path"))
        feature = self.feature_var.get()
        value = row.get(feature, "")
        meta = (
            f"row_id={row.get('row_id')} | component_id={row.get('component_id', '')} | "
            f"group={row.get('group')} | camera={row.get('camera_mode')} | "
            f"defect_type={row.get('defect_type')} | {feature}={self.format_number(value)}"
        )
        if image_path is None:
            self.image_viewer.show_error(f"{meta}\n\n이미지 파일을 찾지 못했습니다: {row.get('image_path')}")
            return
        self.image_viewer.update(image_path, meta)

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
