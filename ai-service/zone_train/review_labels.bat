@echo off
REM ============================================================
REM  Skinglow - open the zone label review page (review_tool.py)
REM  Run from the folder that has zone_dataset and dataset_clean_v2.
REM  Round 1 (test/valid, no AI):  review_labels.bat
REM  Round 2 (train, AI hints):    review_labels.bat --model runs_zone\<run>\best.pt
REM ============================================================
setlocal
set KIT=%~dp0
if "%DATA_ROOT%"=="" set DATA_ROOT=%CD%\dataset_clean_v2
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python "%KIT%review_tool.py" --zones zone_dataset --data-root "%DATA_ROOT%" --reviews zone_reviews.jsonl %*
