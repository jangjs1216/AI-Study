# CSV schema for `side_row_col_visualizer.py`

CSV 파일은 헤더 1줄과 데이터 행으로 구성되어야 합니다.

필수 의미 컬럼:

- 날짜 컬럼: 예) `Date`, `Datetime`, `Timestamp`, `날짜`, `일자`
- Side 컬럼: `BL`, `BT`, `BR`, `BB`만 집계합니다.
- Row 컬럼: 각 Side 내 행 번호
- Col 컬럼: 각 Side 내 열 번호

예시:

```csv
Date,Side,Row,Col
2026-05-01,BL,0,0
2026-05-01,BLeft,30,1
2026-05-01,BTop,1,20
2026-05-02,BB,2,1
```

격자 크기:

- `BL`, `BR`: Row `0~30`, Col `0~1` 범위의 31 x 2 격자
- `BT`, `BB`: 2 x 20

실행:

```powershell
python .\side_row_col_visualizer.py
```

프로그램에서 CSV를 선택한 뒤 컬럼 매핑을 확인하고, 날짜 구간을 입력하면 됩니다.
