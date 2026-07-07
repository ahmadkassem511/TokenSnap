@echo off
setlocal
REM ============================================================
REM  Tokensnap installer for Windows
REM  Checks Python, creates a virtual environment, installs
REM  tokensnap, and shows how to get it on PATH.
REM ============================================================
cd /d "%~dp0"

echo.
echo === Tokensnap v2 installer ===
echo.

REM --- 1. Find Python ---------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo.
    echo Install Python 3.9+ first, e.g.:
    echo     winget install Python.Python.3.12
    echo or download it from https://www.python.org/downloads/
    echo Then re-run this installer.
    echo.
    pause
    exit /b 1
)

REM --- 2. Check version >= 3.9 -------------------------------
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.9 or newer is required.
    python --version
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo Found %%v

REM --- 3. Create virtual environment -------------------------
if not exist ".venv" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

REM --- 4. Install tokensnap ----------------------------------
echo Installing tokensnap ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
    echo [ERROR] pip install failed. See the output above.
    pause
    exit /b 1
)

echo.
echo === Installed successfully! ===
echo.
echo The tokensnap command lives at:
echo     %CD%\.venv\Scripts\tokensnap.exe
echo.
set /p ADDPATH="Add tokensnap to your user PATH? [y/N] "
if /i "%ADDPATH%"=="y" (
    for /f "usebackq tokens=2,*" %%A in (`reg query "HKCU\Environment" /v Path 2^>nul`) do set "USERPATH=%%B"
    setx PATH "%USERPATH%;%CD%\.venv\Scripts" >nul
    echo Added. Open a NEW terminal for PATH changes to take effect.
) else (
    echo Skipped. You can run it via the full path above, or activate the venv:
    echo     .venv\Scripts\activate
)

echo.
set /p OPENDASH="Do you want to open the setup dashboard now? [Y/n] "
if /i not "%OPENDASH%"=="n" (
    echo Starting the dashboard in the background - it will open in your browser...
    start "" ".venv\Scripts\tokensnap.exe" dashboard
) else (
    echo Skipped. Start it later with: tokensnap dashboard
)

echo.
set /p MAKESHORTCUT="Create a desktop shortcut to open the dashboard? [Y/n] "
if /i not "%MAKESHORTCUT%"=="n" (
    for /f "delims=" %%D in ('powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"') do set "DESKTOPDIR=%%D"
    powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $lnk = $ws.CreateShortcut('%DESKTOPDIR%\TokenSnap Dashboard.lnk'); $lnk.TargetPath = '%CD%\.venv\Scripts\tokensnap.exe'; $lnk.Arguments = 'dashboard'; $lnk.WorkingDirectory = '%CD%'; $lnk.WindowStyle = 7; $lnk.IconLocation = '%CD%\.venv\Scripts\tokensnap.exe,0'; $lnk.Description = 'Open the TokenSnap dashboard'; $lnk.Save()"
    if exist "%DESKTOPDIR%\TokenSnap Dashboard.lnk" (
        echo Desktop shortcut created: "TokenSnap Dashboard.lnk"
    ) else (
        echo [WARN] Could not create the desktop shortcut. You can still run: tokensnap dashboard
    )
) else (
    echo Skipped desktop shortcut.
)

echo.
echo Quickstart:
echo   tokensnap dashboard        (web UI: setup wizard, charts ^& settings)
echo   tokensnap start            (start the proxy)
echo   tokensnap run claude       (launch Claude Code through the proxy)
echo   tokensnap monitor          (live savings dashboard, terminal)
echo   tokensnap preset smart     (activate intelligent selective compression)
echo.
echo For best quality: run 'tokensnap preset smart' once.
echo Optionally, get a free OpenRouter key ^(https://openrouter.ai/keys^)
echo to enable AI-powered Memory Cards: tokensnap config set openrouter_api_key YOUR_KEY
echo.
pause