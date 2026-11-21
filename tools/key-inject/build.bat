@echo off
rem Build key_inject.exe against the migrated ao host/.
setlocal

call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul
if errorlevel 1 ( echo failed to set up VS environment & exit /b 1 )

cd /d "%~dp0"

set "AO_HOST=..\..\..\ao\host"

ml64 /nologo /c "%AO_HOST%\ao_call.asm" /Fo ao_call.obj
if errorlevel 1 exit /b 1

cl /nologo /O2 /W3 /I "%AO_HOST%" key_inject.c ao_call.obj ^
   /link user32.lib ^
   /out:key_inject.exe >build.log 2>&1
set RC=%ERRORLEVEL%

if %RC% NEQ 0 (
    echo === key_inject build FAILED ===
    type build.log
    exit /b 1
)

echo OK
dir /b key_inject.exe
exit /b 0
