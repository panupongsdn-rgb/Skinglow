@echo off
REM ============================================================
REM  Skinglow - Train / Validation / Test accuracy of the deployed model
REM  eval_splits.bat [runs_zone\<run>]   (default: newest run, retuned\best.pt)
REM  Result: <run>\retuned\splits\  (accuracy_splits.md / .json / .png)
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
set CKPT=%RUN%\retuned\best.pt
if not exist "%CKPT%" set CKPT=%RUN%\best.pt
set TEST_REVIEWS=zone_reviews.jsonl
if exist review_compare\zone_reviews_final.jsonl set TEST_REVIEWS=review_compare\zone_reviews_final.jsonl
echo Model: %CKPT%
%PY% "%KIT%eval_splits.py" --ckpt "%CKPT%" --data zone_dataset --valid-reviews zone_reviews.jsonl --test-reviews "%TEST_REVIEWS%"
if errorlevel 1 (echo FAILED - see the error above & exit /b 1)
for %%f in ("%CKPT%") do start "" "%%~dpfsplits"
