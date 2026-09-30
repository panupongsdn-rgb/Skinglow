@echo off
REM ============================================================
REM  Skinglow - open the zone label review page (review_tool.py)
REM  Run from the folder that has zone_dataset (or zone_dataset_test) and dataset_clean_v2.
REM  Round 1 (test/valid, no AI):  review_labels.bat
REM  Round 2 (train, AI hints):    review_labels.bat --model runs_zone\<run>\best.pt
REM ============================================================
setlocal
set KIT=%~dp0
set PY=python
if exist "%CD%\.venv\Scripts\python.exe" set PY="%CD%\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" set PY="%KIT%..\..\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" cd /d "%KIT%..\.."
if "%DATA_ROOT%"=="" set DATA_ROOT=%CD%\dataset_clean_v2
if "%ZONES%"=="" if exist zone_dataset\faces.jsonl set ZONES=zone_dataset
if "%ZONES%"=="" set ZONES=zone_dataset_test
%PY% "%KIT%review_tool.py" --zones "%ZONES%" --data-root "%DATA_ROOT%" --reviews zone_reviews.jsonl %*
