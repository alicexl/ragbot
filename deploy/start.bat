@echo off
REM 前台启动（先跑通再上 NSSM 服务化，见 README「部署」节）
cd /d %~dp0..
call deploy\env.bat
"D:\Anaconda3\python.exe" -X utf8 -m app.main
