@echo off
rem Build osu_overlay.exe against the migrated ao host/.
setlocal

call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul
if errorlevel 1 ( echo failed to set up VS environment & exit /b 1 )

cd /d "%~dp0"

set "AO_HOST=..\..\..\ao\host"

ml64 /nologo /c "%AO_HOST%\ao_call.asm" /Fo ao_call.obj
if errorlevel 1 exit /b 1

cl /nologo /O2 /W3 /I "%AO_HOST%" osu_overlay.c ao_call.obj ^
   /link psapi.lib user32.lib ^
   /out:osu_overlay.exe >build.log 2>&1
set RC=%ERRORLEVEL%

if %RC% NEQ 0 (
    echo === osu_overlay build FAILED ===
    type build.log
    exit /b 1
)

echo OK
dir /b osu_overlay.exe
exit /b 0
