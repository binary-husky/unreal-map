choice /t 3 /d y /n >nul
start /wait package1.bat
start /wait package2.bat
start /wait ./../pack_linux.bat
pause