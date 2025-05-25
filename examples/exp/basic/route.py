# vps_server.py
from easyremote import Server

# Start the gateway server
server = Server(port=8080)
server.start()
