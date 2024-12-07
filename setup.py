# setup.py
import os
import subprocess
from setuptools import setup, find_packages
from distutils.command.build_py import build_py

class BuildProtoCommand(build_py):
    def run(self):
        proto_file = os.path.join("easyremote", "protos", "service.proto")
        proto_dir = os.path.dirname(proto_file)
        
        os.makedirs(proto_dir, exist_ok=True)
        
        subprocess.check_call([
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file
        ])
        
        # 修复导入路径
        pb2_file = os.path.join(proto_dir, "service_pb2_grpc.py")
        with open(pb2_file, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'import service_pb2 as service__pb2',
            'from . import service_pb2 as service__pb2'
        )
        
        with open(pb2_file, 'w') as f:
            f.write(content)
            
        super().run()

setup(
    name="easyremote",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.51.0",
        "protobuf>=4.21.0",
        "grpcio-tools>=1.51.0",
    ],
    cmdclass={
        'build_py': BuildProtoCommand,
    },
)