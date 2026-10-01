@echo off
REM ============================================================
REM  Skinglow - compare two independent reviews and settle disagreements
REM  Run from the folder that has zone_dataset_test and dataset_clean_v2.
REM    compare_reviews.bat            -> report in review_compare\agreement.md
REM    compare_reviews.bat adjudicate -> open the page with only the disagreements
REM    compare_reviews.bat final      -> review_compare\zone_reviews_final.jsonl
REM  A = zone_reviews.jsonl (you), B = zone_reviews_claude.jsonl (second reviewer)
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
if "%A%"=="" set A=zone_reviews.jsonl
if "%B%"=="" set B=zone_reviews_claude.jsonl
if /I "%1"=="adjudicate" (
  %PY% "%KIT%review_tool.py" --zones "%ZONES%" --data-root "%DATA_ROOT%" --compare "%A%" "%B%" --reviews zone_reviews_adjudicated.jsonl
) else if /I "%1"=="final" (
  %PY% "%KIT%compare_reviews.py" --a "%A%" --b "%B%" --adjudicated zone_reviews_adjudicated.jsonl --final
) else (
  %PY% "%KIT%compare_reviews.py" --a "%A%" --b "%B%"
)
