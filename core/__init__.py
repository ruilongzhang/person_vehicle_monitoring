from person_vehicle_monitoring.tools import httpclient
from person_vehicle_monitoring.config import TRITON_HTTP_SERVER_URL

CLIENT = httpclient.InferenceServerClient(url='172.30.2.18:9911', verbose=False)
CLIENT_DECORATE = httpclient.InferenceServerClient(url='172.30.2.18:9922', verbose=False)
