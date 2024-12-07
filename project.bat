@echo off
rem 创建根目录
mkdir easyremote
cd easyremote

rem 创建根目录下的文件
echo. 2>.gitignore
echo. 2> LICENSE
echo. 2> README.md
echo. 2> pyproject.toml
echo. 2> setup.py
echo. 2> requirements.txt

rem 创建examples目录及里面的文件
mkdir examples
cd examples
echo. 2> compute_node.py
echo. 2> flask_server.py
echo. 2> fastapi_server.py
cd ..

rem 创建easyremote主代码目录及里面的文件
mkdir easyremote
cd easyremote
echo. 2> __init__.py
echo. 2> server.py
echo. 2> compute_node.py
echo. 2> decorators.py
echo. 2> exceptions.py
echo. 2> types.py
echo. 2> utils.py

rem 创建easyremote主代码目录下的protos目录及里面的文件
mkdir protos
cd protos
echo. 2> __init__.py
echo. 2> service.proto
cd ..
cd ..

rem 创建tests目录及里面的文件
mkdir tests
cd tests
echo. 2> __init__.py
echo. 2> test_server.py
echo. 2> test_compute_node.py
echo. 2> test_decorators.py
cd ..