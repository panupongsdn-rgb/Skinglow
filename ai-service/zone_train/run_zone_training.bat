@echo off
REM ============================================================
REM  Skinglow - build the zone dataset, train, and evaluate
REM  Usage (from the folder that has .venv and dataset_clean_v2):
REM     path\to\Skinglow\ai-service\zone_train\run_zone_training.bat
REM  Optional: set ARCH / EPOCHS / BATCH before running, e.g.
REM     set ARCH=efficientnet_b2& set BATCH=32& run_zone_training.bat
REM ============================================================
setlocal
set KIT=%~dp0
set PY=python
if exist "%CD%\.venv\Scripts\python.exe" set PY="%CD%\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" set PY="%KIT%..\..\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" cd /d "%KIT%..\.."
if "%DATA_ROOT%"=="" set DATA_ROOT=%CD%\dataset_clean_v2
if "%ARCH%"=="" set ARCH=efficientnet_b0
if "%EPOCHS%"=="" set EPOCHS=40
if "%BATCH%"=="" set BATCH=64

set REVIEW_ARGS=
if exist zone_reviews.jsonl set REVIEW_ARGS=--reviews zone_reviews.jsonl --eval-reviewed-only

if not exist zone_dataset\labels.csv (
    echo [1/2] Building zone dataset from %DATA_ROOT% ...
    %PY% "%KIT%build_zone_dataset.py" --data-root "%DATA_ROOT%" --out zone_dataset || goto :error
) else (
    echo [1/2] zone_dataset\labels.csv already exists - delete the folder to rebuild
)

echo [2/2] Training %ARCH% ...
%PY% "%KIT%train_zone_classifier.py" --data zone_dataset --arch %ARCH% --epochs %EPOCHS% --batch %BATCH% --workers 4 %REVIEW_ARGS% || goto :error

echo.
echo Done. Open runs_zone\...\report_test.md for the Accuracy / F1 tables.
goto :eof

:error
echo.
echo FAILED - see the error above.
exit /b 1
