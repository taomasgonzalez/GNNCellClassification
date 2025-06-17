import os
import signal
import subprocess
import time


def start_mlflow_server(port, artifacts_dir, pid_file_path):

    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./artifacts",
        "--host", "0.0.0.0",
        "--port", port,
    ]

    os.makedirs(artifacts_dir, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    with open(pid_file_path, "w") as f:
        f.write(str(proc.pid))

    time.sleep(1.0)
    print(f"MLflow server started (PID {proc.pid})")


def stop_mlflow_server(pid_file_path):
    with open(pid_file_path) as f:
        pid = int(f.read().strip())

    os.kill(pid, signal.SIGTERM)
    os.remove(pid_file_path)

    print(f"MLflow server stopped (PID {pid})")
