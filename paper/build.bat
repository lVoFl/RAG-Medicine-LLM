@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo Building main.tex
echo ========================================

latexmk -xelatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo.
    echo Build failed. Please check the log above.
    pause
    exit /b 1
)

if not exist "build" mkdir build

copy /y "main.pdf" "build\main.pdf" >nul
if exist "main.synctex.gz" copy /y "main.synctex.gz" "build\main.synctex.gz" >nul

echo.
echo ========================================
echo Build finished
echo PDF: build\main.pdf
echo ========================================

pause
exit /b 0
