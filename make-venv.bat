@echo off
if .%1. == .. (set VENV=venv) else (set VENV=%1)
if .%2. == .. (set REQ_TXT=requirements.txt) else (set REQ_TXT=%2)

set SETUP_FILE=c:\dev\python-3.11.9-amd64.exe
set SYSTEM_PYTHONHOME=C:\dev\python-3.11

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR_ENCODED=!PROJECT_DIR:\=_!"
set "PROJECT_DIR_ENCODED=!PROJECT_DIR_ENCODED::=!"

if not exist !VENV! (
    echo Creating !VENV!
    if not exist !SYSTEM_PYTHONHOME! (
        echo ^> !SETUP_FILE! /quiet InstallAllUsers=1 PrependPath=1 TargetDir=!SYSTEM_PYTHONHOME!
        !SETUP_FILE! /quiet InstallAllUsers=1 PrependPath=1 TargetDir=!SYSTEM_PYTHONHOME!
    )

    echo ^> !SYSTEM_PYTHONHOME!\python -m venv !VENV!
    !SYSTEM_PYTHONHOME!\python -m venv !VENV!

    call !VENV!\Scripts\activate.bat

    timeout 3
    echo python -m pip install --upgrade pip pyinstaller -q
    python -m pip install --upgrade pip pyinstaller -q
) else (
    call !VENV!\Scripts\activate.bat
)

if exist !REQ_TXT! (
    if exist %VIRTUAL_ENV%\Scripts\pip.exe (
		%VIRTUAL_ENV%\Scripts\python -c "import pkg_resources as p, sys as s; m=[str(r) for r in p.parse_requirements(open('requirements.txt')) if not p.working_set.by_key.get(r.key)]; print('\n'.join(m)) if m else None; s.exit(1 if m else 0)" || (
            echo ^> %VIRTUAL_ENV%\Scripts\pip install -r !REQ_TXT! -q
			%VIRTUAL_ENV%\Scripts\pip install -r !REQ_TXT! -q
		)
    )
)
endlocal & chcp 65001 > NUL 2>&1 & call %VENV%\Scripts\activate
