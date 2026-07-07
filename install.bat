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
echo Creating a desktop shortcut for the dashboard...
REM A generated .ps1 file (rather than one long -Command line) avoids the
REM nested single/double-quote mess that comes from mixing batch variable
REM expansion, PowerShell string literals, and cmd's own quote parsing -
REM that combination previously produced an empty desktop path and tried
REM (and failed, for lack of permission) to save the shortcut to C:\.
REM A space before every >> is required: cmd.exe treats a bare digit
REM immediately adjacent to a redirection operator (e.g. "7>>") as an
REM explicit file-descriptor number, silently swallowing it - which is
REM exactly what corrupted the WindowStyle line before this fix.
set "SHORTCUT_PS1=%TEMP%\tokensnap_make_shortcut.ps1"
echo $desktop = [Environment]::GetFolderPath('Desktop') > "%SHORTCUT_PS1%"
echo $ws = New-Object -ComObject WScript.Shell >> "%SHORTCUT_PS1%"
echo $lnk = $ws.CreateShortcut("$desktop\TokenSnap Dashboard.lnk") >> "%SHORTCUT_PS1%"
echo $lnk.TargetPath = "%CD%\.venv\Scripts\tokensnap.exe" >> "%SHORTCUT_PS1%"
echo $lnk.Arguments = "dashboard" >> "%SHORTCUT_PS1%"
echo $lnk.WorkingDirectory = "%CD%" >> "%SHORTCUT_PS1%"
echo $lnk.WindowStyle = 7 >> "%SHORTCUT_PS1%"
echo $lnk.IconLocation = "%CD%\.venv\Scripts\tokensnap.exe,0" >> "%SHORTCUT_PS1%"
echo $lnk.Description = "Open the TokenSnap dashboard" >> "%SHORTCUT_PS1%"
echo $lnk.Save() >> "%SHORTCUT_PS1%"
echo Write-Output $desktop >> "%SHORTCUT_PS1%"
for /f "delims=" %%D in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%SHORTCUT_PS1%"') do set "DESKTOPDIR=%%D"
del "%SHORTCUT_PS1%" >nul 2>nul
if exist "%DESKTOPDIR%\TokenSnap Dashboard.lnk" (
    echo Desktop shortcut created: "TokenSnap Dashboard.lnk"
) else (
    echo [WARN] Could not create the desktop shortcut. You can still run: tokensnap dashboard
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