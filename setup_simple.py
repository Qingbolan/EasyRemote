# setup_simple.py
from setuptools import setup, find_packages

setup(
    name="easyremote",
    version="0.1.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "grpcio>=1.68.1",
        "protobuf>=5.29.0",
        "uvicorn>=0.32.1",
        "python-multipart>=0.0.19",
        "rich>=13.0.0"
    ],
    author="Silan.Hu",
    author_email="silan.hu@u.nus.edu",
    description="Easy remote function execution framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Qingbolan/easyremote",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 