from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


VALID_SIDES = ("BL", "BT", "BR", "BB")
SIDE_LABELS = {
    "BL": "BL / BLeft",
    "BT": "BT / BTop",
    "BR": "BR / BRight",
    "BB": "BB / BBottom",
}
SIDE_GRID_SHAPES = {
    "BL": (31, 2),
    "BR": (31, 2),
    "BT": (2, 20),
    "BB": (2, 20),
}
SIDE_AUTO_BASES = {
    "BL": (0, 0),
    "BR": (0, 0),
}

ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

COLUMN_CANDIDATES = {
    "date": (
        "date",
        "datetime",
        "timestamp",
        "time",
        "day",
        "createdate",
        "createdat",
        "inspectiondate",
        "reviewdate",
        "workdate",
        "lotdate",
        "날짜",
        "일자",
        "시간",
        "검사일",
        "작업일",
        "발생일",
    ),
    "side": (
        "side",
        "bside",
        "edge",
        "position",
        "region",
        "area",
        "direction",
        "면",
        "방향",
        "위치",
        "영역",
        "부위",
    ),
    "row": (
        "row",
        "r",
        "rowindex",
        "rowidx",
        "rowno",
        "rownum",
        "행",
        "행번호",
    ),
    "col": (
        "col",
        "column",
        "c",
        "colindex",
        "colidx",
        "colno",
        "colnum",
        "columnindex",
        "열",
        "열번호",
    ),
}


@dataclass(frozen=True)
class RawRecord:
    day: date
    side: str
    row_value: int
    col_value: int
    line_number: int


@dataclass(frozen=True)
class NormalizedRecord:
    day: date
    side: str
    row: int
    col: int
    line_number: int


class CsvLoadError(RuntimeError):
    pass


def normalize_header(value: str) -> str:
    return re.sub(r"[\s_\-./()[\]{}:]+", "", value.strip().lower())


def find_column(headers: list[str], kind: str) -> str | None:
    normalized = {header: normalize_header(header) for header in headers}
    candidates = tuple(normalize_header(candidate) for candidate in COLUMN_CANDIDATES[kind])

    for header, name in normalized.items():
        if name in candidates:
            return header

    scored: list[tuple[int, int, str]] = []
    for index, header in enumerate(headers):
        name = normalized[header]
        score = 0
        for candidate in candidates:
            if candidate and candidate in name:
                score = max(score, len(candidate))
            if name and name in candidate:
                score = max(score, max(1, len(name) - 1))
        if score:
            scored.append((score, -index, header))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    last_error: Exception | None = None
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
                    raise CsvLoadError("CSV 헤더를 찾지 못했습니다.")

                headers = [header.strip() for header in reader.fieldnames if header is not None]
                rows: list[dict[str, str]] = []
                for row in reader:
                    clean = {
                        (key.strip() if key is not None else ""): (value.strip() if value is not None else "")
                        for key, value in row.items()
                    }
                    rows.append(clean)
                return headers, rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except OSError as exc:
            raise CsvLoadError(str(exc)) from exc

    raise CsvLoadError(f"CSV 인코딩을 읽지 못했습니다: {last_error}")


def parse_date_value(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None

    excel_serial = parse_int_value(text)
    if excel_serial is not None and 20000 <= excel_serial <= 80000:
        try:
            return date(1899, 12, 30) + timedelta(days=excel_serial)
        except OverflowError:
            pass

    match = re.search(r"(\d{4})\s*[년./-]?\s*(\d{1,2})\s*[월./-]?\s*(\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
    if match:
        month, day, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    for fmt in (
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%m/%d/%Y",
        "%m-%d-%Y",
    ):
        try:
            return datetime.strptime(text[: max(len(text), len(fmt))], fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_int_value(value: str) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
    if not number.is_integer():
        return None
    return int(number)


def normalize_side(value: str) -> str | None:
    text = re.sub(r"[\s_\-]+", "", value.strip().upper())
    aliases = {
        "BL": "BL",
        "BLEFT": "BL",
        "LEFT": "BL",
        "L": "BL",
        "BT": "BT",
        "BTOP": "BT",
        "TOP": "BT",
        "T": "BT",
        "BR": "BR",
        "BRIGHT": "BR",
        "RIGHT": "BR",
        "R": "BR",
        "BB": "BB",
        "BBOTTOM": "BB",
        "BOTTOM": "BB",
        "B": "BB",
    }
    return aliases.get(text)


def parse_date_entry(text: str) -> date:
    parsed = parse_date_value(text)
    if parsed is None:
        raise ValueError("날짜는 YYYY-MM-DD 형식으로 입력해 주세요.")
    return parsed


def to_display_date(value: date | None) -> str:
    return value.isoformat() if value else ""


def build_raw_records(
    rows: list[dict[str, str]],
    date_column: str,
    side_column: str,
    row_column: str,
    col_column: str,
) -> tuple[list[RawRecord], Counter[str]]:
    records: list[RawRecord] = []
    skipped: Counter[str] = Counter()

    for offset, row in enumerate(rows, start=2):
        day = parse_date_value(row.get(date_column, ""))
        if day is None:
            skipped["invalid_date"] += 1
            continue

        side = normalize_side(row.get(side_column, ""))
        if side not in VALID_SIDES:
            skipped["ignored_side"] += 1
            continue

        row_value = parse_int_value(row.get(row_column, ""))
        col_value = parse_int_value(row.get(col_column, ""))
        if row_value is None or col_value is None:
            skipped["invalid_position"] += 1
            continue

        records.append(RawRecord(day=day, side=side, row_value=row_value, col_value=col_value, line_number=offset))

    return records, skipped


def infer_base(values: Iterable[int], limit: int, mode: str) -> int:
    if mode == "0":
        return 0
    if mode == "1":
        return 1

    value_list = list(values)
    if not value_list:
        return 1
    if min(value_list) == 0:
        return 0
    if max(value_list) <= limit:
        return 1
    return 0


def normalize_records(
    records: list[RawRecord],
    row_base_mode: str,
    col_base_mode: str,
) -> tuple[list[NormalizedRecord], Counter[str], dict[str, tuple[int, int]]]:
    normalized: list[NormalizedRecord] = []
    skipped: Counter[str] = Counter()
    inferred_bases: dict[str, tuple[int, int]] = {}

    for side in VALID_SIDES:
        side_records = [record for record in records if record.side == side]
        row_limit, col_limit = SIDE_GRID_SHAPES[side]
        default_row_base, default_col_base = SIDE_AUTO_BASES.get(side, (None, None))
        row_base = (
            default_row_base
            if row_base_mode == "auto" and default_row_base is not None
            else infer_base((record.row_value for record in side_records), row_limit, row_base_mode)
        )
        col_base = (
            default_col_base
            if col_base_mode == "auto" and default_col_base is not None
            else infer_base((record.col_value for record in side_records), col_limit, col_base_mode)
        )
        inferred_bases[side] = (row_base, col_base)

        for record in side_records:
            row_index = record.row_value - row_base
            col_index = record.col_value - col_base
            if not (0 <= row_index < row_limit and 0 <= col_index < col_limit):
                skipped["out_of_grid"] += 1
                continue
            normalized.append(
                NormalizedRecord(
                    day=record.day,
                    side=record.side,
                    row=row_index,
                    col=col_index,
                    line_number=record.line_number,
                )
            )

    normalized.sort(key=lambda item: (item.day, item.side, item.row, item.col))
    return normalized, skipped, inferred_bases


def aggregate_counts(
    records: list[NormalizedRecord],
    start_day: date,
    end_day: date,
) -> dict[str, list[list[int]]]:
    counts = {
        side: [[0 for _ in range(SIDE_GRID_SHAPES[side][1])] for _ in range(SIDE_GRID_SHAPES[side][0])]
        for side in VALID_SIDES
    }
    for record in records:
        if start_day <= record.day <= end_day:
            counts[record.side][record.row][record.col] += 1
    return counts


def interpolate_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    start_values = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_values = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(round(start_values[i] + (end_values[i] - start_values[i]) * ratio) for i in range(3))
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def heat_color(value: int, max_value: int) -> str:
    if value <= 0 or max_value <= 0:
        return "#f8fafc"
    ratio = value / max_value
    if ratio < 0.5:
        return interpolate_color("#dbeafe", "#60a5fa", ratio / 0.5)
    return interpolate_color("#60a5fa", "#dc2626", (ratio - 0.5) / 0.5)


class HeatmapGrid(ttk.Frame):
    def __init__(self, parent: tk.Widget, side: str) -> None:
        super().__init__(parent, padding=(8, 6))
        self.side = side
        self.rows, self.cols = SIDE_GRID_SHAPES[side]
        self.counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.total = 0
        self.max_value = 0
        self.row_label_base = 0
        self.col_label_base = 0

        self.title_var = tk.StringVar(value=SIDE_LABELS[side])
        ttk.Label(self, textvariable=self.title_var, font=("", 11, "bold")).pack(anchor="w")
        canvas_height = max(132, min(560, self.rows * 16 + 48))
        self.canvas = tk.Canvas(self, height=canvas_height, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        self.canvas.pack(fill="x", expand=True, pady=(5, 0))
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def set_counts(self, counts: list[list[int]], row_label_base: int = 0, col_label_base: int = 0) -> None:
        self.counts = counts
        self.row_label_base = row_label_base
        self.col_label_base = col_label_base
        self.total = sum(sum(row) for row in counts)
        self.max_value = max((value for row in counts for value in row), default=0)
        self.title_var.set(f"{SIDE_LABELS[self.side]}    total={self.total}    max cell={self.max_value}")
        self.draw()

    def draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 118)

        left_margin = 42
        right_margin = 10
        top_margin = 24
        bottom_margin = 24
        usable_width = max(1, width - left_margin - right_margin)
        usable_height = max(1, height - top_margin - bottom_margin)
        cell_width = usable_width / self.cols
        cell_height = usable_height / self.rows
        font_size = max(6, min(10, int(cell_width * 0.42), int(cell_height * 0.45)))

        for col in range(self.cols):
            coord = col + self.col_label_base
            if self.cols <= 5 or col == 0 or coord % 5 == 0 or col == self.cols - 1:
                x = left_margin + (col + 0.5) * cell_width
                self.canvas.create_text(x, 11, text=str(coord), fill="#475569", font=("", 8))

        for row in range(self.rows):
            y = top_margin + (row + 0.5) * cell_height
            self.canvas.create_text(20, y, text=f"R{row + self.row_label_base}", fill="#475569", font=("", 8, "bold"))

        for row in range(self.rows):
            for col in range(self.cols):
                value = self.counts[row][col]
                x1 = left_margin + col * cell_width
                y1 = top_margin + row * cell_height
                x2 = left_margin + (col + 1) * cell_width
                y2 = top_margin + (row + 1) * cell_height
                color = heat_color(value, self.max_value)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")
                text_color = "#ffffff" if self.max_value and value / self.max_value >= 0.58 else "#0f172a"
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(value),
                    fill=text_color,
                    font=("", font_size, "bold"),
                )

        self.canvas.create_text(left_margin, height - 10, text="Col", anchor="w", fill="#64748b", font=("", 8))


class SideDistributionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Side Row/Col Distribution Visualizer")
        self.geometry("1160x860")
        self.minsize(980, 720)

        self.csv_path: Path | None = None
        self.headers: list[str] = []
        self.rows: list[dict[str, str]] = []
        self.raw_records: list[RawRecord] = []
        self.records: list[NormalizedRecord] = []
        self.available_dates: list[date] = []
        self.load_skipped: Counter[str] = Counter()
        self.normalize_skipped: Counter[str] = Counter()
        self.inferred_bases: dict[str, tuple[int, int]] = {}

        self.path_var = tk.StringVar(value="CSV 파일을 선택하세요.")
        self.status_var = tk.StringVar(value="대기 중")
        self.mode_var = tk.StringVar(value="day")
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.current_day_var = tk.StringVar()
        self.encoding_var = tk.StringVar()
        self.row_base_var = tk.StringVar(value="auto")
        self.col_base_var = tk.StringVar(value="auto")
        self.column_vars = {kind: tk.StringVar() for kind in ("date", "side", "row", "col")}

        self._build_ui()
        self._set_controls_enabled(False)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        file_frame = ttk.Frame(root)
        file_frame.pack(fill="x")
        ttk.Button(file_frame, text="CSV 열기", command=self.open_csv).pack(side="left")
        ttk.Label(file_frame, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=10)

        mapping = ttk.LabelFrame(root, text="컬럼 매핑", padding=8)
        mapping.pack(fill="x", pady=(10, 0))
        for index, (kind, label) in enumerate((("date", "Date"), ("side", "Side"), ("row", "Row"), ("col", "Col"))):
            ttk.Label(mapping, text=label).grid(row=0, column=index * 2, sticky="w", padx=(0, 4))
            ttk.Combobox(
                mapping,
                textvariable=self.column_vars[kind],
                width=22,
                state="readonly",
                values=(),
            ).grid(row=0, column=index * 2 + 1, sticky="ew", padx=(0, 12))
        mapping.columnconfigure(1, weight=1)
        mapping.columnconfigure(3, weight=1)
        mapping.columnconfigure(5, weight=1)
        mapping.columnconfigure(7, weight=1)

        base_frame = ttk.Frame(mapping)
        base_frame.grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Label(base_frame, text="Row 기준").pack(side="left")
        ttk.Combobox(base_frame, textvariable=self.row_base_var, width=8, state="readonly", values=("auto", "0", "1")).pack(
            side="left", padx=(4, 14)
        )
        ttk.Label(base_frame, text="Col 기준").pack(side="left")
        ttk.Combobox(base_frame, textvariable=self.col_base_var, width=8, state="readonly", values=("auto", "0", "1")).pack(
            side="left", padx=(4, 14)
        )
        ttk.Button(base_frame, text="매핑 적용", command=self.apply_mapping).pack(side="left")
        ttk.Label(base_frame, textvariable=self.encoding_var).pack(side="left", padx=(14, 0))

        range_frame = ttk.LabelFrame(root, text="날짜 제어", padding=8)
        range_frame.pack(fill="x", pady=(10, 0))
        ttk.Radiobutton(range_frame, text="현재 날짜", variable=self.mode_var, value="day", command=self.refresh_visuals).pack(
            side="left"
        )
        ttk.Radiobutton(range_frame, text="구간 합계", variable=self.mode_var, value="range", command=self.refresh_visuals).pack(
            side="left", padx=(4, 12)
        )
        ttk.Label(range_frame, text="시작").pack(side="left")
        ttk.Entry(range_frame, textvariable=self.start_var, width=12).pack(side="left", padx=(4, 10))
        ttk.Label(range_frame, text="끝").pack(side="left")
        ttk.Entry(range_frame, textvariable=self.end_var, width=12).pack(side="left", padx=(4, 10))
        ttk.Button(range_frame, text="구간 적용", command=self.apply_date_range).pack(side="left", padx=(0, 16))
        ttk.Button(range_frame, text="< 하루", command=lambda: self.shift_day(-1)).pack(side="left")
        ttk.Entry(range_frame, textvariable=self.current_day_var, width=12, justify="center").pack(side="left", padx=4)
        ttk.Button(range_frame, text="하루 >", command=lambda: self.shift_day(1)).pack(side="left")

        summary_frame = ttk.Frame(root)
        summary_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(summary_frame, textvariable=self.status_var).pack(anchor="w")

        grid_frame = ttk.Frame(root)
        grid_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.grids = {side: HeatmapGrid(grid_frame, side) for side in VALID_SIDES}
        placements = {
            "BL": (0, 0),
            "BR": (0, 1),
            "BT": (1, 0),
            "BB": (1, 1),
        }
        for row_index, weight in enumerate((4, 1)):
            grid_frame.rowconfigure(row_index, weight=weight)
        for column_index in range(2):
            grid_frame.columnconfigure(column_index, weight=1)
        for side, (row_index, column_index) in placements.items():
            self.grids[side].grid(row=row_index, column=column_index, sticky="nsew", padx=(0, 8), pady=(0, 8))

    def _mapping_comboboxes(self) -> list[ttk.Combobox]:
        comboboxes: list[ttk.Combobox] = []
        for widget in self.winfo_children():
            comboboxes.extend(self._find_comboboxes(widget))
        return comboboxes

    def _find_comboboxes(self, widget: tk.Widget) -> list[ttk.Combobox]:
        found: list[ttk.Combobox] = []
        if isinstance(widget, ttk.Combobox):
            found.append(widget)
        for child in widget.winfo_children():
            found.extend(self._find_comboboxes(child))
        return found

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        readonly = "readonly" if enabled else "disabled"
        for widget in self.winfo_children():
            self._set_child_state(widget, state, readonly)

    def _set_child_state(self, widget: tk.Widget, state: str, readonly: str) -> None:
        if isinstance(widget, ttk.Button):
            if widget["text"] != "CSV 열기":
                widget.configure(state=state)
        elif isinstance(widget, (ttk.Entry, ttk.Radiobutton)):
            widget.configure(state=state)
        elif isinstance(widget, ttk.Combobox):
            widget.configure(state=readonly)
        for child in widget.winfo_children():
            self._set_child_state(child, state, readonly)

    def open_csv(self) -> None:
        file_name = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=(("CSV files", "*.csv"), ("Text files", "*.txt;*.tsv"), ("All files", "*.*")),
            initialdir=str(Path.cwd()),
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            headers, rows, encoding = read_csv_rows(path)
        except CsvLoadError as exc:
            messagebox.showerror("CSV 읽기 실패", str(exc))
            return

        self.csv_path = path
        self.headers = headers
        self.rows = rows
        self.raw_records = []
        self.records = []
        self.path_var.set(str(path))
        self.encoding_var.set(f"encoding={encoding}, rows={len(rows):,}")

        for combobox in self._mapping_comboboxes():
            values = tuple(headers)
            if combobox.cget("width") == 22:
                combobox.configure(values=values)

        for kind in ("date", "side", "row", "col"):
            detected = find_column(headers, kind)
            self.column_vars[kind].set(detected or (headers[0] if headers else ""))

        self._set_controls_enabled(True)
        self.apply_mapping()

    def apply_mapping(self) -> None:
        if not self.rows:
            return

        columns = {kind: self.column_vars[kind].get() for kind in ("date", "side", "row", "col")}
        missing = [kind for kind, column in columns.items() if not column]
        if missing:
            messagebox.showwarning("컬럼 매핑 필요", f"컬럼을 선택해 주세요: {', '.join(missing)}")
            return

        raw_records, skipped = build_raw_records(
            self.rows,
            date_column=columns["date"],
            side_column=columns["side"],
            row_column=columns["row"],
            col_column=columns["col"],
        )
        normalized, normalize_skipped, inferred_bases = normalize_records(
            raw_records,
            row_base_mode=self.row_base_var.get(),
            col_base_mode=self.col_base_var.get(),
        )

        self.raw_records = raw_records
        self.records = normalized
        self.load_skipped = skipped
        self.normalize_skipped = normalize_skipped
        self.inferred_bases = inferred_bases
        self.available_dates = sorted({record.day for record in self.records})

        if not self.available_dates:
            self.start_var.set("")
            self.end_var.set("")
            self.current_day_var.set("")
            self.refresh_visuals()
            messagebox.showwarning("집계 데이터 없음", "BL, BT, BR, BB에 해당하는 유효 데이터가 없습니다.")
            return

        start_day = self.available_dates[0]
        end_day = self.available_dates[-1]
        self.start_var.set(to_display_date(start_day))
        self.end_var.set(to_display_date(end_day))
        self.current_day_var.set(to_display_date(start_day))
        self.refresh_visuals()

    def apply_date_range(self) -> None:
        try:
            start_day = parse_date_entry(self.start_var.get())
            end_day = parse_date_entry(self.end_var.get())
        except ValueError as exc:
            messagebox.showwarning("날짜 입력 오류", str(exc))
            return
        if start_day > end_day:
            messagebox.showwarning("날짜 입력 오류", "시작 날짜가 끝 날짜보다 늦습니다.")
            return

        current = parse_date_value(self.current_day_var.get()) or start_day
        if current < start_day:
            current = start_day
        elif current > end_day:
            current = end_day
        self.current_day_var.set(to_display_date(current))
        self.refresh_visuals()

    def shift_day(self, offset: int) -> None:
        if not self.records:
            return
        try:
            start_day = parse_date_entry(self.start_var.get())
            end_day = parse_date_entry(self.end_var.get())
        except ValueError as exc:
            messagebox.showwarning("날짜 입력 오류", str(exc))
            return

        current = parse_date_value(self.current_day_var.get()) or start_day
        next_day = current + timedelta(days=offset)
        if next_day < start_day:
            next_day = start_day
        elif next_day > end_day:
            next_day = end_day
        self.current_day_var.set(to_display_date(next_day))
        self.mode_var.set("day")
        self.refresh_visuals()

    def selected_period(self) -> tuple[date, date] | None:
        if not self.records:
            return None
        try:
            start_day = parse_date_entry(self.start_var.get())
            end_day = parse_date_entry(self.end_var.get())
        except ValueError:
            return None
        if self.mode_var.get() == "day":
            current = parse_date_value(self.current_day_var.get()) or start_day
            return current, current
        return start_day, end_day

    def refresh_visuals(self) -> None:
        period = self.selected_period()
        if period is None:
            empty_counts = aggregate_counts([], date(2000, 1, 1), date(2000, 1, 1))
            for side in VALID_SIDES:
                row_base, col_base = SIDE_AUTO_BASES.get(side, (0, 0))
                self.grids[side].set_counts(empty_counts[side], row_base, col_base)
            self.status_var.set("표시할 데이터가 없습니다.")
            return

        start_day, end_day = period
        counts = aggregate_counts(self.records, start_day, end_day)
        for side in VALID_SIDES:
            row_base, col_base = self.inferred_bases.get(side, SIDE_AUTO_BASES.get(side, (0, 0)))
            self.grids[side].set_counts(counts[side], row_base, col_base)

        period_label = start_day.isoformat() if start_day == end_day else f"{start_day.isoformat()} ~ {end_day.isoformat()}"
        total = sum(sum(sum(row) for row in side_counts) for side_counts in counts.values())
        side_totals = ", ".join(f"{side}={sum(sum(row) for row in counts[side])}" for side in VALID_SIDES)
        skipped_parts = []
        skipped_parts.extend(f"{key}={value}" for key, value in sorted(self.load_skipped.items()) if value)
        skipped_parts.extend(f"{key}={value}" for key, value in sorted(self.normalize_skipped.items()) if value)
        skipped_text = f" | skipped: {', '.join(skipped_parts)}" if skipped_parts else ""
        base_text = self._base_summary()
        self.status_var.set(f"{period_label} | total={total:,} | {side_totals}{base_text}{skipped_text}")

    def _base_summary(self) -> str:
        if not self.inferred_bases:
            return ""
        values = sorted(set(self.inferred_bases.values()))
        if len(values) == 1:
            row_base, col_base = values[0]
            return f" | index base row={row_base}, col={col_base}"
        parts = ", ".join(f"{side}:R{base[0]}/C{base[1]}" for side, base in self.inferred_bases.items())
        return f" | index base {parts}"


def run_self_test() -> None:
    sample_rows = [
        {"Date": "2026-05-01", "Side": "BL", "Row": "0", "Col": "0"},
        {"Date": "2026-05-01 12:30:00", "Side": "BLeft", "Row": "30", "Col": "1"},
        {"Date": "2026/05/02", "Side": "BT", "Row": "1", "Col": "20"},
        {"Date": "20260502", "Side": "BBottom", "Row": "2", "Col": "1"},
        {"Date": "2026.05.02", "Side": "XX", "Row": "1", "Col": "1"},
    ]
    raw, skipped = build_raw_records(sample_rows, "Date", "Side", "Row", "Col")
    normalized, out_skipped, _bases = normalize_records(raw, "auto", "auto")
    counts = aggregate_counts(normalized, date(2026, 5, 1), date(2026, 5, 2))
    assert skipped["ignored_side"] == 1
    assert out_skipped["out_of_grid"] == 0
    assert counts["BL"][0][0] == 1
    assert counts["BL"][30][1] == 1
    assert counts["BT"][0][19] == 1
    assert counts["BB"][1][0] == 1
    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize Side/Row/Col distribution from CSV.")
    parser.add_argument("--self-test", action="store_true", help="Run parser and aggregation checks without opening UI.")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    os.chdir(Path(__file__).resolve().parent)
    app = SideDistributionApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
