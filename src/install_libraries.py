import subprocess
import sys
import os
  
if sys.platform == "win32":
    path_scripts = os.path.dirname(sys.executable) + os.sep + "Scripts" + os.sep 
    cwd = os.getcwd()
    os.chdir(path_scripts)

subprocess.run(['pip3', 'install', 'numpy'])
subprocess.run(['pip3', 'install', 'matplotlib'])
subprocess.run(['pip3', 'install', 'adjustText'])
subprocess.run(['pip3', 'install', 'nbconvert'])
subprocess.run(['pip3', 'install', 'pandas'])
subprocess.run(['pip3', 'install', 'seaborn'])
subprocess.run(['pip3', 'install', 'scipy'])
subprocess.run(['pip3', 'install', 'scikit-learn'])
subprocess.run(['pip3', 'install', 'statsmodels'])
subprocess.run(['pip3', 'install', 'xgboost'])
subprocess.run(['pip3', 'install', 'openpyxl'])
# subprocess.run('pip3 install --force-reinstall -v "openpyxl==3.1.0"')
subprocess.run(['pip3', 'install', 'zipfile'])
subprocess.run(['pip3', 'install', 'torch'])
subprocess.run(['pip3', 'install', 'torchvision'])
subprocess.run(['pip3', 'install', 'torchaudio'])
subprocess.run(['pip3', 'install', 'threadpoolctl'])

if sys.platform == "win32":
    os.chdir(cwd)