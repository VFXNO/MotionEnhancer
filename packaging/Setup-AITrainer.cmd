@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\ai-train\Setup-AITrainer.ps1" -Pause -Launch %*
exit /b %errorlevel%
