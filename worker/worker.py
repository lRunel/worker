import os
import sys
import time
import uuid
import requests
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import threading
import queue

# Add parent dir to path to allow importing from common/models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import get_controller_url
from common.serialization import (
    decode_state_dict, 
    decode_tensor, 
    decode_gradients,
    encode_sparse_gradients,
    sparsify_gradients
)
from models import get_model
from trainer import Trainer

# ── ANSI Helpers ──────────────────────────────────────────────────────────────

CLEAR      = "\033[2J"
HOME       = "\033[H"
ERASE_LINE = "\033[2K"
HIDE       = "\033[?25l"
SHOW       = "\033[?25h"
BOLD       = "\033[1m"
DIM        = "\033[2m"
RESET      = "\033[0m"
GREEN      = "\033[32m"
CYAN       = "\033[36m"
YELLOW     = "\033[33m"
RED        = "\033[31m"
WHITE      = "\033[37m"

_ANSI_RE   = __import__('re').compile(r'\033\[[0-9;]*m')
_prev_lines = [0]  # mutable container to track previous frame line count



def _render_worker(state: dict):
    """Render a compact worker status panel."""
    W = 56
    inner = W - 4

    lines = []

    def hr_top():
        lines.append(f"  {CYAN}╔{'═' * (W - 2)}╗{RESET}")
    def hr_mid():
        lines.append(f"  {CYAN}╠{'═' * (W - 2)}╣{RESET}")
    def hr_bot():
        lines.append(f"  {CYAN}╚{'═' * (W - 2)}╝{RESET}")
    def row(text: str):
        plain = _ANSI_RE.sub('', text)
        pad = inner - len(plain)
        if pad < 0: pad = 0
        lines.append(f"  {CYAN}║{RESET} {text}{' ' * pad} {CYAN}║{RESET}")

    hr_top()
    title = "Distributed DL — Worker"
    pad_l = (inner - len(title)) // 2
    pad_r = inner - len(title) - pad_l
    lines.append(f"  {CYAN}║{RESET}{' ' * pad_l}{BOLD}{WHITE}{title}{RESET}{' ' * pad_r}  {CYAN}║{RESET}")
    hr_mid()

    # Connection
    status = state.get("status", "connecting")
    if status == "connected":
        color = GREEN
        icon = "●"
    elif status == "training":
        color = GREEN
        icon = "●"
    else:
        color = YELLOW
        icon = "○"

    short_id = state.get("worker_id", "—")[:8]
    row(f"{DIM}Worker{RESET}   {BOLD}{short_id}{RESET}  {color}{icon} {status}{RESET}")
    row(f"{DIM}Type{RESET}     {state.get('device_type', '—'):<10} {DIM}Cores{RESET} {state.get('cpu_cores', '—')}")
    row(f"{DIM}Server{RESET}   {state.get('server', '—')}")
    hr_mid()

    # Training stats
    row(f"{BOLD}TRAINING{RESET}")
    model = state.get("model_name", "—")
    ver   = state.get("model_version", -1)
    row(f"  {DIM}Model{RESET}      {model}  {DIM}v{RESET}{ver}")

    tasks = state.get("tasks_done", 0)
    sps   = state.get("samples_per_sec", 0)
    sps_str = f"{sps:.0f} samp/s" if sps > 0 else "—"
    row(f"  {DIM}Tasks{RESET}      {tasks:<8} {DIM}Speed{RESET}  {sps_str}")

    bw = state.get("bandwidth", 0)
    if bw > 1_000_000:
        bw_str = f"{bw / 1_000_000:.1f} MB/s"
    elif bw > 1000:
        bw_str = f"{bw / 1000:.1f} KB/s"
    else:
        bw_str = "—"
    row(f"  {DIM}Bandwidth{RESET}  {bw_str}")

    # Current action
    action = state.get("action", "Idle")
    action_color = YELLOW if "Training" in action else CYAN if "Fetching" in action else GREEN if "Sending" in action else DIM
    row(f"  {DIM}Status{RESET}     {action_color}{action}{RESET}")

    hr_bot()

    # ── Output: single contiguous string, no stray newlines ──
    out = HIDE
    if _prev_lines[0] > 0:
        out += f"\033[{_prev_lines[0]}A\r"
    for i, line in enumerate(lines):
        if i > 0:
            out += "\n"
        out += f"{ERASE_LINE}{line}"
    extra = _prev_lines[0] - len(lines)
    if extra > 0:
        for _ in range(extra):
            out += f"\n{ERASE_LINE}"
        out += f"\033[{extra}A"
    out += "\n"

    sys.stderr.write(out)
    sys.stderr.flush()
    _prev_lines[0] = len(lines)


class WorkerNode:
    def __init__(self, controller_ip="127.0.0.1", controller_port=8000):
        self.worker_id = str(uuid.uuid4())
        self.controller_url = get_controller_url(controller_ip, controller_port)
        
        # System specs
        if HAS_PSUTIL:
            self.cpu_cores = psutil.cpu_count(logical=True)
            self.ram_mb = int(psutil.virtual_memory().total / (1024**2))
        else:
            self.cpu_cores = os.cpu_count() or 4
            self.ram_mb = 4096  # Assume 4GB as a safe fallback for mobile
            
        # Try to infer device type
        if 'TERMUX_VERSION' in os.environ:
            self.device_type = "phone"
        else:
            self.device_type = "laptop"
            
        # Stats tracking
        self.samples_per_sec = 0.0
        self.bandwidth = 0.0 # bytes/sec
        
        # Local state tracking for Delta sync
        self.fetch_version = -1
        self.model_version = -1
        
        # Model state
        self.model = None
        self.model_name = None
        self.model_config = None
        self.model_state = None

        # Dashboard state
        self._tasks_done = 0
        self._ui_state = {
            "worker_id": self.worker_id,
            "device_type": self.device_type,
            "cpu_cores": self.cpu_cores,
            "server": f"{controller_ip}:{controller_port}",
            "status": "connecting",
            "model_name": "—",
            "model_version": -1,
            "tasks_done": 0,
            "samples_per_sec": 0,
            "bandwidth": 0,
            "action": "Connecting…",
        }

    def _render(self):
        _render_worker(self._ui_state)

    def register(self):
        payload = {
            "worker_id": self.worker_id,
            "cpu_cores": self.cpu_cores,
            "ram": self.ram_mb,
            "device_type": self.device_type
        }
        self._ui_state["action"] = "Registering…"
        self._render()
        try:
            r = requests.post(f"{self.controller_url}/register", json=payload)
            r.raise_for_status()
            self._ui_state["status"] = "connected"
            self._ui_state["action"] = "Registered ✓"
            self._render()
            return True
        except requests.exceptions.RequestException:
            self._ui_state["action"] = "Registration failed — retrying"
            self._render()
            return False

    def fetch_task_loop(self, task_queue):
        """Background thread that continuously fetches the next task."""
        while True:
            # wait if queue is full
            if task_queue.qsize() >= 2:
                time.sleep(0.1)
                continue
                
            try:
                params = {
                    "worker_id": self.worker_id,
                    "samples_per_sec": self.samples_per_sec,
                    "bandwidth": self.bandwidth,
                    "model_version": self.fetch_version
                }
                
                req_start_time = time.time()
                r = requests.get(f"{self.controller_url}/get_task", params=params)
                r.raise_for_status()
                
                download_time = time.time() - req_start_time
                payload_size = len(r.content)
                self.bandwidth = payload_size / max(download_time, 0.001)
                self._ui_state["bandwidth"] = self.bandwidth
                
                task = r.json()
                if "model_version" in task:
                    self.fetch_version = task["model_version"]
                    
                task_queue.put(task)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400 and "complete" in e.response.text:
                    task_queue.put({"type": "DONE"})
                    break
                time.sleep(2)
            except requests.exceptions.RequestException:
                self._ui_state["action"] = "Connection lost — retrying"
                self._render()
                time.sleep(2)
            except Exception:
                time.sleep(2)

    def run(self):
        # Register first
        while not self.register():
            time.sleep(5)
            
        self._ui_state["status"] = "training"
        self._ui_state["action"] = "Waiting for first task…"
        self._render()
        
        task_queue = queue.Queue(maxsize=2)
        fetch_thread = threading.Thread(target=self.fetch_task_loop, args=(task_queue,), daemon=True)
        fetch_thread.start()
        while True:
            try:
                # 1. Get Task from Queue (blocks until available)
                task = task_queue.get()
                if "type" in task and task["type"] == "DONE":
                    self._ui_state["action"] = "Training complete ✓"
                    self._ui_state["status"] = "done"
                    self._render()
                    sys.stderr.write(SHOW)
                    sys.stderr.flush()
                    break
                    
                update_type = task.get("model_update_type", "full")
                sparsity_ratio = task.get("sparsity_ratio", 1.0)
                
                if update_type == "full":
                    self.model_name   = task["model_name"]
                    self.model_config = task["model_config"]
                    self.model_state  = decode_state_dict(task["model_state"])
                    self.model_version = task["model_version"]
                    self.model = get_model(self.model_name, self.model_config)
                    self.model.load_state_dict(self.model_state)
                elif update_type == "delta":
                    deltas = decode_gradients(task["delta_gradients"])
                    if deltas is not None and self.model is not None:
                        for param, d in zip(self.model.parameters(), deltas):
                            param.data.add_(d.to(param.device))
                        self.model_state = self.model.state_dict()
                    self.model_version = task["model_version"]
                elif update_type == "none":
                    pass # model state is already up to date
                
                data         = decode_tensor(task["data"])
                labels       = decode_tensor(task["labels"])
                steps        = task["steps"]
                lr           = task.get("learning_rate", 0.001)

                self._ui_state["model_name"] = self.model_name or "—"
                self._ui_state["model_version"] = self.model_version
                self._ui_state["action"] = f"Training v{self.model_version} ({steps} steps)"
                self._render()
                
                # 2. Local Training — model-agnostic
                trainer = Trainer(self.model_name, self.model_config, self.model_state, lr=lr)
                
                train_start_time = time.time()
                gradients = trainer.train_steps(data, labels, steps)
                train_time = time.time() - train_start_time
                
                self.samples_per_sec = len(data) / max(train_time, 0.001)
                self._tasks_done += 1

                self._ui_state["samples_per_sec"] = self.samples_per_sec
                self._ui_state["tasks_done"] = self._tasks_done
                self._ui_state["action"] = f"Sending gradients (s={sparsity_ratio:.2f})"
                self._render()
                
                # 3. Send Gradients (Fire and forget via background thread to overlap with next fetch/compute)
                sparse_grads = sparsify_gradients(gradients, sparsity_ratio)
                send_payload = {
                    "worker_id": self.worker_id,
                    "gradients": encode_sparse_gradients(sparse_grads),
                    "model_version": self.model_version,
                    "samples_processed": len(data)
                }
                
                # Send in a separate daemon thread so computation on the next task starts instantly
                def send_async(payload):
                    try:
                        requests.post(f"{self.controller_url}/send_gradients", json=payload)
                    except Exception:
                        pass
                        
                threading.Thread(target=send_async, args=(send_payload,), daemon=True).start()
                
            except Exception:
                self._ui_state["action"] = "Error — retrying"
                self._render()
                time.sleep(2)

if __name__ == "__main__":
    # If run standalone
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Deep Learning Worker")
    parser.add_argument("--controller-ip", default="127.0.0.1", help="Controller node IP address")
    parser.add_argument("--controller-port", type=int, default=8000, help="Controller node port")
    args = parser.parse_args()
    
    worker = WorkerNode(controller_ip=args.controller_ip, controller_port=args.controller_port)
    worker.run()
