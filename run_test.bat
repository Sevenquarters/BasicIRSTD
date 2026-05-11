@echo off
cd /d "d:\Program Files (x86)\IRSTD\BasicIRSTD"
call venv\Scripts\activate.bat
python test_resunet_cbam.py
pause
