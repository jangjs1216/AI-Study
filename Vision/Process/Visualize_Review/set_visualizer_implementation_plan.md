# set_visualizer 구현 계획

## 목표

`set_visualizer.py`는 CSV를 읽는 프로그램이 아니라, 사용자가 지정한 4개 이미지 루트 경로에서 날짜, 장비/모델 정보, 패치 이미지를 수집해 스마트폰 전개도 UI에 표시하는 프로그램으로 만든다.

표시 대상 면은 다음 6개다.

- `A`: 정면 또는 기준면
- `BL`: 왼쪽 옆면
- `BT`: 위쪽 옆면
- `BB`: 아래쪽 옆면
- `BR`: 오른쪽 옆면
- `C`: `BR` 오른쪽에 배치되는 `A`와 같은 크기의 면

현재 UI의 면 비율은 유지한다.

- `A`, `C`: `20 x 31`
- `BL`, `BR`: `2 x 31`
- `BT`, `BB`: `20 x 2`

## 전체 흐름

1. 사용자가 `A`, `BL/BR`, `BT/BB`, `C`에 해당하는 4개 루트 경로를 지정한다.
2. 사용자가 `완료` 버튼을 누르면, 각 루트 경로 바로 아래의 날짜 폴더를 스캔한다.
3. 날짜 폴더명은 `260520` 같은 `yymmdd` 형식으로 해석한다.
4. UI에서 날짜 폴더를 선택하고 `적용` 버튼을 누른다.
5. 선택한 날짜에 대해 4개 루트의 날짜 폴더 내부를 스캔한다.
6. 날짜 폴더 내부에는 검사 단위 폴더가 있으며, 폴더명에서 `hhmmss`, `IMEI`, `모델파일`, `ColorCode`를 수집한다.
7. 각 검사 단위 폴더 안의 이미지 파일을 수집한다.
8. 이미지 파일명에서 `row`, `col`, `Type`, `Vector`를 추출한다.
9. `BL/BR`, `BT/BB` 루트에서는 `Type` 값으로 실제 face를 판별한다.
10. 추출한 패치를 IMEI별, 면별, 카테고리별로 집계한다.
11. UI는 다음 두 가지 방식으로 표시할 수 있어야 한다.
    - 전체 패치 수를 면별 patch grid에 표시
    - IMEI별 defect 발생 개수를 기준으로 표시

## 경로 구조 이해

사용자가 지정하는 루트 경로는 4개다.

- `A`: A면 전용 루트
- `BL/BR`: BL, BR이 함께 들어 있는 루트
- `BT/BB`: BT, BB가 함께 들어 있는 루트
- `C`: C면 전용 루트

예상 구조:

```text
<face_root>
  260528
    122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW
      Defects[0][9][C_Center][0].png
      Refined[1][1][B_Top][0].png
      Filtered[0][9][C_Center][0].png
```

예시 경로:

```text
\\0.0.0.125\FinalCosmetic\260528\122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW\Defects[0][9][C_Center][0].png
\\0.0.0.125\FinalCosmetic\260528\122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW\Refined[1][1][B_Top][0].png
\\0.0.0.125\FinalCosmetic\260528\122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW\Filtered[0][9][C_Center][0].png
```

주의할 점:

- 예시 기준으로는 `Defects`, `Refined`, `Filtered`가 하위 폴더명이 아니라 파일명 prefix처럼 보인다.
- 실제 구조가 `...\Defects\[0][9][C_Center][0].png` 형태일 수도 있으므로, 구현 시 두 구조를 모두 지원할 수 있게 설계한다.

## 날짜 폴더 파싱

날짜 폴더명 규칙:

```text
yymmdd
```

예:

```text
260520
260528
```

검증 규칙:

- 정규식: `^\d{6}$`
- `datetime.strptime(name, "%y%m%d")`로 실제 날짜인지 검증한다.
- UI 표시값은 원본 폴더명과 해석된 날짜를 같이 보여줄 수 있다.
  - 예: `260528 (2026-05-28)`

날짜 목록 병합 방식:

- 4개 루트 경로에서 발견한 날짜 폴더의 합집합을 UI에 표시한다.
- 선택한 날짜가 일부 루트에 없으면, 해당 루트는 누락 상태로 표시하고 스캔은 가능한 루트만 진행한다.

## 검사 단위 폴더명 파싱

검사 단위 폴더명 예:

```text
122850_ABC1234567890_SM-S948-SMART_COSMETIC_V26.03.10.0_ZW
```

수집해야 하는 값:

- `hhmmss`: `122850`
- `IMEI`: `ABC1234567890`
- `모델파일`: `SM-S948-SMART_COSMETIC_V26.03.10.0`
- `ColorCode`: `ZW`

중요 조건:

- 모델파일에는 underscore가 들어갈 수 있다.
- 따라서 단순히 `_`로 전체 split하면 안 된다.
- 앞에서 `hhmmss`, `IMEI`를 먼저 가져오고, 뒤에서 `ColorCode`를 가져온 뒤, 그 사이 전체를 모델파일로 해석한다.

권장 파싱 방식:

```python
head, color_code = folder_name.rsplit("_", 1)
hhmmss, imei, model_file = head.split("_", 2)
```

검증 규칙:

- `hhmmss`: `^\d{6}$`
- `ColorCode`: 길이 2
- `IMEI`: 비어 있지 않아야 함
- `model_file`: 비어 있지 않아야 함

## 이미지 파일명 파싱

사용자가 설명한 이미지 형태:

```text
[row][col][Type][Vector].png
```

예시 파일명:

```text
Defects[0][9][C_Center][0].png
Refined[1][1][B_Top][0].png
Filtered[0][9][C_Center][0].png
```

`Refined`, `Filtered`에는 다음 확장 형태도 존재한다.

```text
[yymmdd][imei][x][y][FacePart][PositionPart][Vector].png
```

예:

```text
[260528][ABC1234567890][0][23][BRight][Center][0].png
```

이 형태는 `FacePart`와 `PositionPart`를 합쳐 기존 `Type`과 같은 값으로 정규화한다.

```text
BRight + Center -> BRight_Center
```

수집해야 하는 값:

- `category`: `Defects`, `Refined`, `Filtered`
- `x`: `0`, `1` 등
- `y`: `9`, `1` 등
- `type`: `C_Center`, `BLeft_Bottom`, `BRight_Top`, `BTop_Right`, `BBottom_Left` 등
- `vector`: `0` 등

권장 정규식:

```text
^(?P<category>[^\[]+)\[(?P<row>\d+)\]\[(?P<col>\d+)\]\[(?P<type>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$
```

확장 형태 정규식:

```text
^\[(?P<file_day>\d{6})\]\[(?P<file_imei>[^\]]+)\]\[(?P<x>\d+)\]\[(?P<y>\d+)\]\[(?P<face_part>[^\]]+)\]\[(?P<position_part>[^\]]+)\]\[(?P<vector>[^\]]+)\]\.png$
```

대소문자 처리:

- 확장자는 `.png`, `.PNG` 모두 허용한다.
- category는 원본 문자열을 유지하되, 내부 비교용으로 소문자 정규화 값을 별도로 둔다.

face 판별 규칙:

- `A`, `C` 루트는 루트 슬롯 자체가 face 기준이다.
- `BL/BR` 루트는 `Type` 값의 prefix로 face를 결정한다.
  - `BLeft_*` 또는 `B_Left_*`: `BL`
  - `BRight_*` 또는 `B_Right_*`: `BR`
- `BT/BB` 루트는 `Type` 값의 prefix로 face를 결정한다.
  - `BTop_*` 또는 `B_Top_*`: `BT`
  - `BBottom_*` 또는 `B_Bottom_*`: `BB`
- 루트 슬롯과 `Type` prefix가 맞지 않는 파일은 `patch type root mismatch`로 집계하고 표시 대상에서 제외한다.

원본 patch 참조:

- `Defects`, `Refined`, `Filtered` patch를 수집할 때 같은 검사 폴더의 sibling인 `Patches` 폴더에 있는 원본 이미지 경로도 내부 메타데이터로 함께 보관한다.
- `Patches` 폴더는 탐색하지 않는다.
- 현재 patch 메타데이터로 아래 파일명을 O(1)로 구성한다.

```text
<inspection_folder>/Patches/[yymmdd][imei][x][y][FacePart][PositionPart][Vector].png
```

- 이 경로는 나중에 viewer에서 원본과 결과 이미지를 비교할 때 사용한다.
- 파일 존재 여부 확인도 scan 단계에서는 수행하지 않는다.

## 데이터 모델

구현 시 다음 정도의 내부 모델을 둔다.

```python
FaceRoot:
    face: str
    root_path: Path

DateFolder:
    face: str
    folder_name: str
    day: date
    path: Path

InspectionFolder:
    face: str
    day_folder: str
    hhmmss: str
    imei: str
    model_file: str
    color_code: str
    path: Path

PatchImage:
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
    original_patch: PatchFileRef | None

PatchFileRef:
    category: str
    row: int
    col: int
    defect_type: str
    vector: str
    path: Path
```

## 집계 방식

기본 patch grid 크기:

```python
GRID_SHAPES = {
    "A": (31, 20),
    "C": (31, 20),
    "BL": (31, 2),
    "BR": (31, 2),
    "BT": (2, 21),
    "BB": (2, 21),
}
```

주의:

- UI 면 비율 표기는 `width x height` 기준이다.
- grid 배열은 기존 `side_row_col_visualizer.py`처럼 `rows x cols`로 둔다.
- 그래서 `A`, `C`는 `31 rows x 20 cols`가 된다.
- 파일명 좌표는 모든 face에서 `[x][y]`로 해석한다.
  - UI grid에는 항상 `[row=y][col=x]`로 변환한다.
  - `A`, `C`: 파일명 `[0~19][0~30]`을 UI `31 rows x 20 cols`로 변환한다.
  - `BL`, `BR`: 파일명 `[0~1][0~30]`을 UI `31 rows x 2 cols`로 변환한다.
  - `BT`, `BB`: 파일명 `[0~20][0~1]`을 UI `2 rows x 21 cols`로 변환한다.

집계 모드:

1. 전체 패치 수 모드
   - 선택한 날짜, 선택한 category 조건에 맞는 모든 이미지 패치를 cell별로 카운트한다.
   - 같은 IMEI에서 같은 row/col 이미지가 여러 개 있으면 모두 카운트한다.

2. IMEI별 defect 수 모드
   - category가 `Defects`인 패치를 기준으로 한다.
   - IMEI별 defect 총 개수를 목록이나 테이블로 표시한다.
   - 선택한 IMEI 하나를 고르면 해당 IMEI의 patch만 grid에 표시할 수 있게 확장한다.

3. 고유 IMEI 발생 수 모드
   - cell별로 defect가 한 번이라도 발생한 고유 IMEI 수를 표시한다.
   - 같은 IMEI가 같은 cell에 여러 이미지를 갖더라도 1로 계산한다.

## UI 계획

상단 설정 영역:

- `A` 경로 선택
- `BL/BR` 경로 선택
- `BT/BB` 경로 선택
- `C` 경로 선택
- `완료` 버튼
- `완료` 버튼을 누르면 4개 root 경로를 사용자 설정 파일에 저장한다.
- 프로그램 시작 시 저장된 root 경로가 있으면 입력칸에 자동 복원하고 날짜 목록을 자동 로드한다.

날짜/필터 영역:

- 왼쪽 비교 패널 날짜 폴더 선택 combobox
- 왼쪽 비교 패널 `적용` 버튼
- 오른쪽 비교 패널 날짜 폴더 선택 combobox
- 오른쪽 비교 패널 `적용` 버튼
- category 필터
  - `All`
  - `Defects`
  - `Refined`
  - `Filtered`
- 표시 모드 선택
  - `전체 패치`
  - `IMEI별 defect 수`
  - `고유 IMEI 발생 수`
- IMEI 선택 combobox
  - 전체 또는 특정 IMEI 선택 가능

중앙 시각화 영역:

- 동일한 전개도 UI를 좌/우 2개 배치한다.
- 각 전개도는 독립적으로 선택된 날짜의 결과를 표시한다.
- category, 표시 모드, IMEI 필터는 두 전개도에 공통 적용한다.
- 기존 전개도 배치 유지
  - `BT`는 `A` 위
  - `BB`는 `A` 아래
  - `BL`, `A`, `BR`, `C`는 가로 배치
- 각 면은 해당 grid cell로 나뉜다.
- cell에는 count를 표시한다.
- count에 따라 heatmap 색상을 적용한다.

하단 상태 영역:

- 선택 날짜
- 스캔한 면 수
- 검사 폴더 수
- IMEI 수
- 이미지 파일 수
- 파싱 실패 수
- grid 범위 밖 row/col 수

## 처리 알고리즘

### 1. 루트 경로 확정

사용자가 4개 경로를 지정하고 `완료`를 누르면:

1. 각 경로 존재 여부를 확인한다.
2. 존재하지 않는 경로는 경고한다.
3. 존재하는 경로에서 `^\d{6}$` 날짜 폴더를 찾는다.
4. 날짜 목록을 병합해 UI에 표시한다.
5. root 경로 4개를 로컬 설정 파일에 저장한다.

설정 파일 위치:

- Windows: `%APPDATA%/SetVisualizer/config.json`
- 그 외 또는 `APPDATA`가 없을 때: `~/.set_visualizer/config.json`

### 2. 날짜 적용

사용자가 날짜를 선택하고 `적용`을 누르면:

1. 각 face의 `<face_root>/<selected_yymmdd>`를 확인한다.
2. 해당 날짜 폴더가 있는 face만 스캔한다.
3. 날짜 폴더 내부의 검사 단위 폴더를 찾는다.
4. 검사 단위 폴더명을 파싱한다.
5. 검사 단위 폴더 안의 이미지 파일을 찾는다.
6. 이미지 파일명을 파싱한다.
7. `PatchImage` 목록을 만든다.
8. 선택된 표시 모드와 필터에 맞춰 grid counts를 만든다.
9. canvas를 다시 그린다.

### 3. 중복 탐색 및 병렬화

`Apply` 동작은 다음 성능 전략을 따른다.

1. 루트 경로 4개와 선택 날짜를 묶은 cache key를 만든다.
2. 동일한 cache key의 결과가 이미 있으면 파일 시스템을 다시 보지 않고 캐시된 `ScanResult`를 사용한다.
3. 동일한 cache key가 이미 스캔 중이면 새 스캔을 만들지 않고 기존 스캔 완료를 기다린다.
4. 새 스캔이 필요한 경우 UI thread가 아니라 background thread에서 실행한다.
5. background scan 내부에서는 검사 폴더 단위 작업을 thread pool에 분배한다.
6. `Defects`, `Refined`, `Filtered` 폴더는 현재 구조상 이미지가 바로 들어 있다고 보고 recursive scan 대신 직접 하위 파일만 조회한다.

### 4. 패치 표시

각 face panel은 다음 순서로 그린다.

1. face 외곽 직사각형
2. row/col grid
3. cell별 heatmap 색
4. cell별 count text
5. face label

## 에러 및 예외 처리

다음 항목은 status 영역 또는 별도 summary에 표시한다.

- 존재하지 않는 루트 경로
- 날짜 폴더명 파싱 실패
- 검사 단위 폴더명 파싱 실패
- 이미지 파일명 파싱 실패
- 루트 슬롯과 `Type` prefix 불일치
- row/col이 해당 face grid 범위를 벗어난 이미지
- 이미지 확장자가 `.png`가 아닌 파일
- 선택 날짜가 특정 face에는 없는 경우

## 구현 순서

1. 현재 `set_visualizer.py`의 UI를 경로 입력 4개와 날짜 선택 흐름으로 확장한다.
2. 날짜 폴더 스캔 함수 작성
   - `scan_date_folders(root_paths) -> list[DateFolder]`
3. 검사 단위 폴더명 파서 작성
   - `parse_inspection_folder_name(name) -> InspectionMeta | None`
4. 이미지 파일명 파서 작성
   - `parse_patch_filename(name) -> PatchMeta | None`
5. 선택 날짜 스캔 함수 작성
   - `scan_patches(root_paths, selected_yymmdd) -> ScanResult`
6. patch 집계 함수 작성
   - 전체 패치 수
   - IMEI별 defect 수
   - 고유 IMEI 발생 수
7. canvas face panel을 grid heatmap 렌더링 방식으로 확장한다.
8. self-test 추가
   - 날짜 폴더 파싱
   - 검사 단위 폴더명 파싱
   - 이미지 파일명 파싱
   - grid 범위 검증
   - 집계 결과 검증

## 확정 사항

1. `Defects`, `Refined`, `Filtered`는 실제 하위 폴더명이다.
   - 구현은 `<inspection_folder>/<category>/[row][col][Type][Vector].png` 구조를 기본으로 한다.
   - 기존 예시처럼 category가 파일명 prefix로 붙은 구조는 보조 호환만 둔다.
2. `A`, `BL/BR`, `BT/BB`, `C`의 각 루트 경로에는 동일한 날짜 폴더 구조가 있다.
3. 이미지 파일의 `row`, `col`은 0-base로 처리한다.
   - 예시가 `[0][9]` 형태이고, grid 범위도 0-base 기준으로 검증한다.
4. `Type` 값은 표시 필터로 쓰지 않고 메타데이터로 보관한다.
   - 단, `BL/BR`, `BT/BB` 공용 루트에서는 `Type` prefix가 실제 face를 결정하는 기준이다.
   - `BLeft`는 `BL`, `BRight`는 `BR`, `BTop`은 `BT`, `BBottom`은 `BB`로 매핑한다.
5. UI에서는 `Defects`, `Refined`, `Filtered` category를 선택할 수 있어야 한다.
   - 전체 패치 표시, IMEI별 count, 고유 IMEI 발생 수 모두 선택 category 기준으로 계산한다.
6. 같은 IMEI, 같은 face, 같은 row/col에 여러 이미지가 있는 경우 category를 분리해서 메타데이터를 수집한다.
   - `Defects`와 `Refined`에 같은 row/col이 있어도 서로 다른 patch record로 보관한다.
   - `All` category 선택 시에는 category가 다른 record를 모두 포함한다.

## 1차 구현 범위

1차 구현에서는 실제 이미지 미리보기 대신 patch count heatmap을 먼저 완성한다.

- 4개 루트 경로 입력
- 날짜 폴더 스캔
- 선택 날짜 적용
- 검사 폴더명 메타데이터 파싱
- `Defects`, `Refined`, `Filtered` 하위 폴더 스캔
- patch 파일명 파싱
- category 필터
- IMEI 필터
- grid 표시 모드
  - `Total patches`: 선택 조건의 patch 개수
  - `Unique IMEI count`: cell별 고유 IMEI 수
- IMEI별 category count 표시는 우선 오른쪽 목록으로 제공한다.
