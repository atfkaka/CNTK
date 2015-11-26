@REM optionally do clean
set ACML_FMA=0
set CYGWIN_BIN=c:\cygwin64\bin

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"

msbuild /p:Platform=x64 /p:Configuration=Debug CNTK.sln
if errorlevel 1 exit /b 1

.\x64\Debug\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop
if errorlevel 1 exit /b 1

msbuild /p:Platform=x64 /p:Configuration=Release CNTK.sln
if errorlevel 1 exit /b 1

.\x64\Release\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop
if errorlevel 1 exit /b 1

set PATH=%PATH%;%CYGWIN_BIN%
python2.7.exe Tests/TestDriver.py run Speech/QuickE2E
if errorlevel 1 exit /b 1
