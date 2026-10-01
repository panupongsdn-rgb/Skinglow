@echo off
REM ============================================================
REM  Skinglow - re-tune thresholds on the human-reviewed VALID set and re-score test
REM  (no retraining).  retune_thresholds.bat [runs_zone\<run>]   (default: newest run)
REM  Result: <run>\retuned\  (best.pt to deploy, reports, figures)
REM ============================================================
setlocal
set KIT=%~dp0
set PY=python
if exist "%CD%\.venv\Scripts\python.exe" set PY="%CD%\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" set PY="%KIT%..\..\.venv\Scripts\python.exe"
if exist "%KIT%..\..\.venv\Scripts\python.exe" cd /d "%KIT%..\.."
set RUN=%~1
if "%RUN%"=="" for /f "delims=" %%d in ('dir /b /ad /o-d runs_zone 2^>nul') do if not defined RUN set RUN=runs_zone\%%d
if "%RUN%"=="" (echo No run found in runs_zone & exit /b 1)
set TEST_REVIEWS=zone_reviews.jsonl
if exist review_compare\zone_reviews_final.jsonl set TEST_REVIEWS=review_compare\zone_reviews_final.jsonl
echo Run: %RUN%   valid labels: zone_reviews.jsonl   test labels: %TEST_REVIEWS%
%PY% "%KIT%retune_thresholds.py" --run "%RUN%" --data zone_dataset --reviews zone_reviews.jsonl --test-reviews "%TEST_REVIEWS%"
if errorlevel 1 (echo FAILED - see the error above & exit /b 1)
start "" "%RUN%\retuned"
