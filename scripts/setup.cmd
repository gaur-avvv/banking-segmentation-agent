@echo off
setlocal
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
endlocal
