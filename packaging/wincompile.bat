RMDIR /S /Q build
RMDIR /S /Q dist
pyinstaller.exe -F MEAPexe.spec
cd  "C:\Program Files (x86)\Inno Setup 5"
ISCC.exe C:\Users\mattc\projects\MEAP\packaging\innosetup\meap.iss
