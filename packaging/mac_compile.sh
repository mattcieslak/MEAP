#!/bin/bash
rm -rf build dist
export ETS_TOOLKIT=qt4
export QT_API=pyqt
pyinstaller MEAPapp.spec
builddmg 
dmgbuild -s build_dmg.py "MEAPInstaller" dist/meap_installer.dmg 