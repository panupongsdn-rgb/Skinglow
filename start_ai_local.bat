@echo off
REM ============================================================
REM  Skinglow - run the AI service on this computer (http://127.0.0.1:8000)
REM  Double-click this file, or run it in cmd. Keep the window open while you use
REM  the website on http://localhost:8080 ; close it (or Ctrl+C) to stop.
REM  First run installs the missing packages (a few minutes, needs internet).
REM ============================================================
setlocal
cd /d "%~dp0ai-service"

if not exist venv\Scripts\python.exe (
    echo [setup] creating Python virtual environment in ai-service\venv ...
    py -3.9 -m venv venv 2>nul || python -m venv venv || goto :error
    venv\Scripts\python -m pip install --upgrade pip
    echo [setup] installing PyTorch CPU ...
    venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :error
)

venv\Scripts\python -c "import mediapipe, ensemble_boxes, torchvision, ultralytics, fastapi, uvicorn, multipart; assert ultralytics.__version__ == '8.4.118'" 2>nul
if errorlevel 1 (
    echo [setup] installing / updating packages from requirements.txt ...
    venv\Scripts\python -m pip install -r requirements.txt || goto :error
)

echo.
echo  Skinglow AI service: http://127.0.0.1:8000   (health check: http://127.0.0.1:8000/health)
echo  Loading models, wait for "Application startup complete" ...
echo.
venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo.
echo FAILED - see the error above.
pause
exit /b 1
