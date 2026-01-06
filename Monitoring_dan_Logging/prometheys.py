from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
import psutil


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"]
)


CPU_USAGE = Gauge(
    "system_cpu_usage",
    "CPU usage percentage"
)

RAM_USAGE = Gauge(
    "system_ram_usage",
    "RAM usage percentage"
)

def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)

# ===== METRICS ENDPOINT =====
def metrics():
    update_system_metrics()
    return Response(generate_latest(), media_type="text/plain")
