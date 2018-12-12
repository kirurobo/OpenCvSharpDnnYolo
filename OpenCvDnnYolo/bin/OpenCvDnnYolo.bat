@echo off
cd /d %~dp0
shift
cd Release
OpenCvDnnYolo.exe %*
cd ..
