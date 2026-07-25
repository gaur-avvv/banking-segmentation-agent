@echo off
setlocal
if /I "%~1"=="--help" goto :help
if /I "%~1"=="-h" goto :help
cd /d "%~dp0.."
if not exist .venv py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,adk]"
if not exist .env copy /Y .env.example .env >nul
set /p DATASET_PATH=Dataset CSV, ZIP, or folder path (leave blank for automatic demo): 
if not "%DATASET_PATH%"=="" (
  findstr /v /b "BANKING_DATA_PATH=" .env > .env.tmp
  echo BANKING_DATA_PATH=%DATASET_PATH%>>.env.tmp
  move /Y .env.tmp .env >nul
)
echo Setup complete.
echo Run: .venv\Scripts\banking-agent.exe setup
echo Run: .venv\Scripts\banking-agent.exe adk-web --port 8001
echo.
echo Choose start:
echo   1) Terminal chat
echo   2) Local API + browser UI
echo   3) Google ADK Web
echo   4) One-shot segmentation query
echo   5) Exit
set /p CHOICE=Selection [5]:
if "%CHOICE%"=="1" .venv\Scripts\banking-agent.exe chat
if "%CHOICE%"=="2" .venv\Scripts\banking-agent.exe api --port 8000
if "%CHOICE%"=="3" .venv\Scripts\banking-agent.exe adk-web --port 8001
if "%CHOICE%"=="4" .venv\Scripts\banking-agent.exe run --query "Segment customers into priority, regular, and dormant groups"
endlocal
exit /b 0
:help
echo Usage: scripts\setup.cmd [--help]
echo Installs the project, asks for a dataset path, and offers a start menu.
echo Example: scripts\setup.cmd
exit /b 0
