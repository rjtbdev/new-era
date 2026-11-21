@echo off
rem Build osu_menu.exe against the migrated ao host/.
setlocal

call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul
if errorlevel 1 ( echo failed to set up VS environment & exit /b 1 )

cd /d "%~dp0"

set "AO_HOST=..\..\..\ao\host"

rem Rebuild ao_call.obj alongside this exe so paths stay simple.
ml64 /nologo /c "%AO_HOST%\ao_call.asm" /Fo ao_call.obj
if errorlevel 1 exit /b 1

cl /nologo /O2 /W3 /I "%AO_HOST%" osu_menu.c ao_call.obj ^
   /link psapi.lib user32.lib ws2_32.lib bcrypt.lib ^
   /out:osu_menu.exe >build.log 2>&1
set RC=%ERRORLEVEL%

if %RC% NEQ 0 (
    echo === osu_menu build FAILED ===
    type build.log
    exit /b 1
)

echo OK
dir /b osu_menu.exe
exit /b 0
