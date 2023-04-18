@echo off
:agn
set versions=
Setlocal enabledelayedexpansion
echo Current installed 3DSlicer version:
set /A V= 1
for /f "tokens=4 delims=\" %%i in ('reg query "HKEY_CURRENT_USER\SOFTWARE\NA-MIC"') do (
set ver=%%i
echo !V!: !ver!
set /A V+= 1
set versions=!versions!;!ver!
)

set /p input=Please select the target version:
if not defined input (echo.illegal input!&goto :agn)
echo.%input%|findstr /i /v "^[0-9]*$" >nul&&(echo.illegal input!&goto :agn)
if %input% GEQ !V! (
echo.illegal input!
goto :agn
)
if %input% LEQ 0 (
echo.illegal input!
goto :agn
)

for /f "tokens=%input% delims=;" %%i in ("%versions%") do (
set ver=%%i
for /f "tokens=1,2,* " %%i IN ('REG QUERY "HKEY_CURRENT_USER\SOFTWARE\NA-MIC\!ver!" /ve') DO SET "SlicerPath=%%k"
)

for /D /R "%SlicerPath%" %%a in ("lib\Slicer*") do (set targetsourcepath=%%a\qt-scripted-modules)

SET Source=%~dp0qt-scripted-modules
SET RequireTxt=%~dp0requirements.txt
SET SlicerPython=%SlicerPath%\bin\PythonSlicer.exe
SET ModelPath=%targetsourcepath%\SegmentCalcDir\model\
XCOPY "%Source%" "%targetsourcepath%" /E /Y /Q
XCOPY "model\" "%ModelPath%"  /E /Y /Q
SET PARAM=-m pip install -r "%RequireTxt%" -i https://pypi.tuna.tsinghua.edu.cn/simple
"%SlicerPython%" %PARAM%

PAUSE