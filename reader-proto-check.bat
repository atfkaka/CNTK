@echo off
setlocal
cd %~dp0
for %%f in (%*) do set a_%%f=1

@REM optionally do clean ?

set ACML_FMA=0
set CYGWIN_BIN=c:\cygwin64\bin
if not exist %CYGWIN_BIN% (
    set CYGWIN_BIN=c:\cygwin\bin
    if not exist %CYGWIN_BIN% (
        echo Can't find Cygwin, is it installed?
        exit /b 1
    )
)
echo on

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
if errorlevel 1 exit /b 1

if not defined a_nodebug (
    msbuild /m /p:Platform=x64 /p:Configuration=Debug CNTK.sln
    if errorlevel 1 exit /b 1

    .\x64\Debug\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop
    if errorlevel 1 exit /b 1
)

if not defined a_norelease (
    msbuild /m /p:Platform=x64 /p:Configuration=Release CNTK.sln
    if errorlevel 1 exit /b 1

    .\x64\Release\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop
    if errorlevel 1 exit /b 1
)

set PATH=%PATH%;%CYGWIN_BIN%

if not defined a_noe2e (
if not defined a_norelease (
if not defined a_nogpu (
    python2.7.exe Tests/TestDriver.py run -d gpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/TestDriver.py run -d cpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

if not defined a_nodebug (
if not defined a_nogpu (
    python2.7.exe Tests/TestDriver.py run -d gpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/TestDriver.py run -d cpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)
)
