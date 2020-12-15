from person_vehicle_monitoring.config import TRITON_SERVER_URL
from person_vehicle_monitoring.tools import httpclient

try:
    http_triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
except Exception as e:
    raise e

