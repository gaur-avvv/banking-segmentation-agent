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
echo Choose provider: 1=Gemini 2=OpenRouter 3=Groq 4=OpenAI 5=Ollama 6=None
set /p PROVIDER_CHOICE=Provider [1]:
if "%PROVIDER_CHOICE%"=="2" (set PROVIDER=openrouter&set KEY_NAME=OPENROUTER_API_KEY) else if "%PROVIDER_CHOICE%"=="3" (set PROVIDER=groq&set KEY_NAME=GROQ_API_KEY) else if "%PROVIDER_CHOICE%"=="4" (set PROVIDER=openai&set KEY_NAME=OPENAI_API_KEY) else if "%PROVIDER_CHOICE%"=="5" (set PROVIDER=ollama&set KEY_NAME=) else if "%PROVIDER_CHOICE%"=="6" (set PROVIDER=none&set KEY_NAME=) else (set PROVIDER=gemini&set KEY_NAME=GEMINI_API_KEY)
findstr /v /b "LLM_PROVIDER=" .env > .env.tmp
echo LLM_PROVIDER=%PROVIDER%>>.env.tmp
if defined KEY_NAME (
  set /p API_KEY=Enter %KEY_NAME% (input will be visible in CMD; use PowerShell for hidden entry):
  findstr /v /b "%KEY_NAME%=" .env.tmp > .env.tmp2
  echo %KEY_NAME%=%API_KEY%>>.env.tmp2
  move /Y .env.tmp2 .env.tmp >nul
)
move /Y .env.tmp .env >nul
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
