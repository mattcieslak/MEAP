import os
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API']='pyqt'


from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'