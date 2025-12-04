# import ray
# import os

# print("Initializing Ray...")
# # Note: We are not using _plasma_directory here to keep the test simple.
# # This will use the default /dev/shm.
# ray.init(_node_ip_address="127.0.0.1")
# print("✅ Ray Initialized Successfully!")

# @ray.remote
# def f(x):
#     return x * x

# futures = [f.remote(i) for i in range(4)]
# print("Tasks submitted. Getting results...")
# results = ray.get(futures)
# print("✅ Results:", results)

# ray.shutdown()
# print("✅ Ray Shutdown Successfully!")

import ray
import os
import sys
import socket
import time

# --- Print info from the main script ---
print("--- Main Script Info ---")
print(f"Python executable: {sys.executable}")
print(f"Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
print("------------------------\n")

# --- Initialize Ray ---
print("Initializing Ray...")
ray.init(_node_ip_address="127.0.0.1")
print("✅ Ray Initialized Successfully!\n")

@ray.remote
def get_worker_info(x):
    # --- Print info from inside the worker process ---
    # This part is the key diagnostic
    print(f"--- Worker-{x} Info ---")
    print(f"Worker Hostname: {socket.gethostname()}")
    print(f"Worker Python executable: {sys.executable}")
    print(f"Worker Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
    print("----------------------")
    time.sleep(1) # Give a moment for the print to flush
    return x * x

futures = [get_worker_info.remote(i) for i in range(4)]
print("Tasks submitted. Waiting for results...")
results = ray.get(futures)
print("\n✅ Results:", results)

ray.shutdown()
print("✅ Ray Shutdown Successfully!")