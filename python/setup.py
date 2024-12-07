from setuptools import setup, find_packages
import platform
import os

def get_lib_path():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == 'windows':
        return f'core/easyremote_{system}_{machine}.dll'
    elif system == 'darwin':
        return f'core/easyremote_{system}_{machine}.dylib'
    else:
        return f'core/easyremote_{system}_{machine}.so'

setup(
    name='easyremote',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'easyremote': [get_lib_path()],
    },
    install_requires=[
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
    ],
    python_requires='>=3.7',
)