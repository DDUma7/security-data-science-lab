import socket
import threading
from queue import Queue
from datetime import datetime

# =========================
# Configuration
# =========================
TARGET = "127.0.0.1"          # change only for systems you own/are authorized to test
START_PORT = 1
END_PORT = 5000               # use 65535 if you want full scan, but it will take longer
THREADS = 100
TIMEOUT = 0.5
ENABLE_BANNER = True

# =========================
# Shared data
# =========================
port_queue = Queue()
results = []
lock = threading.Lock()

# =========================
# Helpers
# =========================
def get_service_name(port: int) -> str:
    try:
        return socket.getservbyport(port)
    except OSError:
        return "unknown"

def grab_banner(ip: str, port: int) -> str:
    """
    Basic banner grabbing. Works only for some services.
    Safe, simple, and suitable for a lab.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT)
            s.connect((ip, port))

            # Send a very small generic request for common text-based services
            try:
                s.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
            except OSError:
                pass

            banner = s.recv(1024).decode(errors="ignore").strip()
            return banner[:120] if banner else ""
    except Exception:
        return ""

def scan_port(ip: str, port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT)
            result = s.connect_ex((ip, port))

            if result == 0:
                service = get_service_name(port)
                banner = grab_banner(ip, port) if ENABLE_BANNER else ""

                with lock:
                    results.append({
                        "port": port,
                        "service": service,
                        "banner": banner
                    })
    except Exception:
        pass

def worker() -> None:
    while not port_queue.empty():
        port = port_queue.get()
        scan_port(TARGET, port)
        port_queue.task_done()

# =========================
# Main
# =========================
def main() -> None:
    print("=" * 60)
    print(f"Advanced Python Port Scanner")
    print(f"Target       : {TARGET}")
    print(f"Port range   : {START_PORT}-{END_PORT}")
    print(f"Threads      : {THREADS}")
    print(f"Banner grab  : {ENABLE_BANNER}")
    print("=" * 60)

    start_time = datetime.now()

    for port in range(START_PORT, END_PORT + 1):
        port_queue.put(port)

    thread_list = []
    for _ in range(THREADS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        thread_list.append(t)

    port_queue.join()

    end_time = datetime.now()
    duration = end_time - start_time

    results.sort(key=lambda x: x["port"])

    print("\nOpen Ports Detected:\n")
    if not results:
        print("No open ports found in the specified range.")
    else:
        for item in results:
            print(f"Port {item['port']:<6} Service: {item['service']:<12} Banner: {item['banner']}")

    print(f"\nTotal open ports: {len(results)}")
    print(f"Scan completed in: {duration}")

if __name__ == "__main__":
    main()