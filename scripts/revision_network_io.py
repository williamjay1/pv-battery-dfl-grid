"""Atomic JSON writes with brief Windows reader-sharing retries."""
import time
from network_replay import save_json as _save_json

def save_json(path,value):
 for attempt in range(21):
  try:
   return _save_json(path,value)
  except PermissionError:
   if attempt==20:raise
   time.sleep(.1)
