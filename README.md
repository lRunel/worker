# Distributed DL Worker

This repository contains everything needed to run a worker node for the Distributed Deep Learning framework. 

## Setup

1. Clone this repository on your worker device.
2. Ensure you have Python installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Worker

To start the worker, execute the `worker.py` script and provide the IP address of your controller node.

```bash
python worker/worker.py --controller-ip <CONTROLLER_IP> --controller-port 8000
```

### Note on Hardware
The worker will automatically detect whether it's running on a mobile device (e.g. via Termux) or a laptop/desktop based on environment variables, and it will register with the controller accordingly.

## Structure
- `worker/`: Contains the main `worker.py` loop and `trainer.py` local training logic.
- `models/`: PyTorch model definitions required for the worker to build the local models.
- `common/`: Shared utilities (serialization, config).
