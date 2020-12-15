import os

RECOGNIZE_CLASS_ID = ['car', 'truck']
TRITON_HTTP_SERVER_URL = os.getenv("TRITON_SERVER_URL", default="10.20.5.9:9911")
TRITON_GRPC_SERVER_URL = os.getenv("TRITON_SERVER_URL", default="10.20.5.9:9710")
#CAMERA_LINE_URL = os.getenv("CAMERALINE_URL", default="http://10.20.2.20:32121/cameraLine/query")

CAMERA_LINE_URL = 'http://172.30.2.10:32121/cameraLine/query'
