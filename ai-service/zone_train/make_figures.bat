@echo off
REM ============================================================
REM  Skinglow - draw result figures (confusion matrix, ROC, PR, per class / zone)
REM  for a finished run. New runs draw them automatically.
REM    make_figures.bat                   -> newest run in runs_zone
REM    make_figures.bat runs_zone\<run>   -> that run
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
set REVIEW_ARGS=
if exist zone_reviews.jsonl set REVIEW_ARGS=--reviews zone_reviews.jsonl --eval-reviewed-only
if exist review_compare\zone_reviews_final.jsonl set REVIEW_ARGS=--reviews review_compare\zone_reviews_final.jsonl --eval-reviewed-only
echo Run: %RUN%
%PY% "%KIT%plot_results.py" --run "%RUN%" --data zone_dataset %REVIEW_ARGS%
if errorlevel 1 (echo FAILED - see the error above & exit /b 1)
start "" "%RUN%"
