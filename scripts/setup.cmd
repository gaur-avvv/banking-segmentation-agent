@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."

if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help

where py >nul 2>&1
if not errorlevel 1 set "PYTHON=py"
if not defined PYTHON where python >nul 2>&1
if not defined PYTHON if not errorlevel 1 set "PYTHON=python"
if not defined PYTHON (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

if not exist .venv %PYTHON% -m venv .venv
if errorlevel 1 goto install_error
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,adk]"
if errorlevel 1 goto install_error
if not exist .env copy /Y .env.example .env >nul

set "DATASET_PATH="
set /p "DATASET_PATH=Dataset CSV, ZIP, or folder path (leave blank for automatic demo): "
if not "%DATASET_PATH%"=="" goto save_dataset
goto choose_provider

:save_dataset
findstr /v /b "BANKING_DATA_PATH=" .env > .env.tmp
echo BANKING_DATA_PATH=%DATASET_PATH%>>.env.tmp
move /Y .env.tmp .env >nul

:choose_provider
echo Choose provider: 1=Gemini 2=OpenRouter 3=Groq 4=OpenAI 5=Ollama 6=None
set "PROVIDER_CHOICE="
set /p "PROVIDER_CHOICE=Provider [1]: "
if "%PROVIDER_CHOICE%"=="2" goto provider_openrouter
if "%PROVIDER_CHOICE%"=="3" goto provider_groq
if "%PROVIDER_CHOICE%"=="4" goto provider_openai
if "%PROVIDER_CHOICE%"=="5" goto provider_ollama
if "%PROVIDER_CHOICE%"=="6" goto provider_none
goto provider_gemini

:provider_gemini
set "PROVIDER=gemini"
set "KEY_NAME=GEMINI_API_KEY"
goto save_provider
:provider_openrouter
set "PROVIDER=openrouter"
set "KEY_NAME=OPENROUTER_API_KEY"
goto save_provider
:provider_groq
set "PROVIDER=groq"
set "KEY_NAME=GROQ_API_KEY"
goto save_provider
:provider_openai
set "PROVIDER=openai"
set "KEY_NAME=OPENAI_API_KEY"
goto save_provider
:provider_ollama
set "PROVIDER=ollama"
set "KEY_NAME="
goto save_provider
:provider_none
set "PROVIDER=none"
set "KEY_NAME="

:save_provider
findstr /v /b "LLM_PROVIDER=" .env > .env.tmp
echo LLM_PROVIDER=%PROVIDER%>>.env.tmp
if "%KEY_NAME%"=="" goto replace_env
set "API_KEY="
set /p "API_KEY=Enter %KEY_NAME% (visible in CMD; use PowerShell for hidden input): "
findstr /v /b "%KEY_NAME%=" .env.tmp > .env.tmp2
echo %KEY_NAME%=%API_KEY%>>.env.tmp2
move /Y .env.tmp2 .env.tmp >nul

:replace_env
move /Y .env.tmp .env >nul
echo Setup complete.
echo.
echo Choose start:
echo   1) Terminal chat
echo   2) Local API + browser UI
echo   3) Google ADK Web
echo   4) One-shot segmentation query
echo   5) Exit
set "CHOICE="
set /p "CHOICE=Selection [5]: "
if "%CHOICE%"=="1" .venv\Scripts\banking-agent.exe chat
if "%CHOICE%"=="2" .venv\Scripts\banking-agent.exe api --port 8000
if "%CHOICE%"=="3" .venv\Scripts\banking-agent.exe adk-web --port 8001
if "%CHOICE%"=="4" .venv\Scripts\banking-agent.exe run --query "Segment customers into priority, regular, and dormant groups"
goto end

:install_error
echo Installation failed. Check Python, pip, and network access.
pause
exit /b 1

:help
echo Usage: scripts\setup.cmd [--help]
echo Installs Python dependencies, asks for dataset/provider settings, and offers a start menu.
echo Requires Python 3.10+ as either py or python on PATH.

:end
endlocal
