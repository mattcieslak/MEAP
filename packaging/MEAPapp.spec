# -*- mode: python -*-
import sys
sys.path.append(".")
from deps2 import imports

block_cipher = None


a = Analysis(['../MEAPapp.py'],
             pathex=['/Users/matt/projects/MEAP/packaging'],
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
          ('resources/logo512x512.png',
          '/Users/matt/projects/MEAP/meap/resources/logo512x512.png','DATA')
          ('resources/meap.png',
          '/Users/matt/projects/MEAP/meap/resources/meap.png','DATA')
]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='MEAPapp',
          debug=True,
          strip=False,
          upx=True,
          console=False , icon='/Users/matt/projects/MEAP/meap/resources/MEAP.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MEAPapp')
               
app = BUNDLE(coll,
             name='MEAP.app',
             bundle_identifier="com.MEAP",
             icon='/Users/matt/projects/MEAP/meap/resources/MEAP.icns')