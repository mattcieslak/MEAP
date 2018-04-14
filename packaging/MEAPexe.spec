# -*- mode: python -*-
import sys
sys.path.append(".")
from deps2 import imports

block_cipher = None
a = Analysis(['..\\MEAPapp.py'],
             pathex=['C:\\Users\\mattc\\projects\\MEAP'],
             binaries=[],
             datas=[],
             hiddenimports=imports,
             hookspath=[],
             runtime_hooks=["rthook_pyqt4.py","rthook_pyface.py"],
             excludes=['matplotlib', 'IPython','FixTk', 'tcl', 'zmq',
                          'tk', '_tkinter', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
             
a.datas += [
          ('resources\\meap.png',
          'C:\\Users\\mattc\\projects\\MEAP\\meap\\resources\\meap.png','DATA'),
          ('resources\\logo512x512.png',
          'C:\\Users\\mattc\\projects\\MEAP\\meap\\resources\\logo512x512.png','DATA'),
          ('srvf_register/dynamic_programming_q2.so',
          'C:\\Users\\mattc\\projects\\srvf_register\\srvf_register\\dynamic_programming_q2.pyd',
          'EXTENSION')
]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# Add the icons
icon_tree = Tree("..\\meap\\resources", prefix="resources")
a.datas += icon_tree


exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='MEAP.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True,
          icon="C:\\Users\\mattc\\projects\\MEAP\\meap\\resources\\meap.ico" )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='MEAP')
