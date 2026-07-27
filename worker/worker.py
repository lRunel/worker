import os
import sys
import time
import uuid
import requests
import psutil
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
from trainer import Trainer

class WorkerNode:
    def __init__(self, controller_ip="127.0.0.1", controller_port=8000):
        self.worker_id = str(uuid.uuid4())
        self.controller_url = get_controller_url(controller_ip, controller_port)
        
        # System specs
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.ram_mb = int(psutil.virtual_memory().total / (1024**2))
        
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
        self.model_state = None
        self.model_name = None
        self.model_config = None
        
    def register(self):
        payload = {
            "worker_id": self.worker_id,
            "cpu_cores": self.cpu_cores,
            "ram": self.ram_mb,
            "device_type": self.device_type
        }
        print(f"[*] Registering worker {self.worker_id}...")
        try:
            r = requests.post(f"{self.controller_url}/register", json=payload)
            r.raise_for_status()
            print("[+] Successfully registered with controller.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"[-] Registration failed: {e}")
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
                
                task = r.json()
                if "model_version" in task:
                    self.fetch_version = task["model_version"]
                    
                task_queue.put(task)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400 and "complete" in e.response.text:
                    print("[i] Training complete. Fetch thread exiting.")
                    task_queue.put({"type": "DONE"})
                    break
                print(f"[!] Fetch HTTP error: {e}")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"[!] Fetch Communication error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[!] Fetch Unexpected error: {e}")
                time.sleep(2)

    def run(self):
        # Register first
        while not self.register():
            print("Retrying registration in 5 seconds...")
            time.sleep(5)
            
        print("[*] Starting training loop...")
        
        task_queue = queue.Queue(maxsize=2)
        fetch_thread = threading.Thread(target=self.fetch_task_loop, args=(task_queue,), daemon=True)
        fetch_thread.start()
        while True:
            try:
                # 1. Get Task from Queue (blocks until available)
                task = task_queue.get()
                if "type" in task and task["type"] == "DONE":
                    break
                    
                update_type = task.get("model_update_type", "full")
                sparsity_ratio = task.get("sparsity_ratio", 1.0)
                
                if update_type == "full":
                    self.model_name   = task["model_name"]
                    self.model_config = task["model_config"]
                    self.model_state  = decode_state_dict(task["model_state"])
                    self.model_version = task["model_version"]
                elif update_type == "delta":
                    deltas = decode_gradients(task["delta_gradients"])
                    for k, d in zip(self.model_state.keys(), deltas):
                        self.model_state[k] = self.model_state[k].cpu() + d.cpu()
                    self.model_version = task["model_version"]
                elif update_type == "none":
                    pass # model state is already up to date
                
                data         = decode_tensor(task["data"])
                labels       = decode_tensor(task["labels"])
                steps        = task["steps"]
                
                print(f"[>] Training task ({self.model_name}, version {self.model_version}): {steps} steps, {len(data)} samples.")
                
                # 2. Local Training — model-agnostic
                trainer = Trainer(self.model_name, self.model_config, self.model_state)
                
                train_start_time = time.time()
                gradients = trainer.train_steps(data, labels, steps)
                train_time = time.time() - train_start_time
                
                self.samples_per_sec = len(data) / max(train_time, 0.001)
                
                # 3. Send Gradients (Fire and forget via background thread to overlap with next fetch/compute)
                print(f"[<] Sending gradients for version {self.model_version} (sparsity {sparsity_ratio:.2f})...")
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
                    except Exception as e:
                        print(f"[-] Failed sending gradients: {e}")
                        
                threading.Thread(target=send_async, args=(send_payload,), daemon=True).start()
                
            except Exception as e:
                print(f"[!] Unexpected error in main loop: {e}")
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
