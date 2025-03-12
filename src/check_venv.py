import os

def is_venv_activated():
    return 'VIRTUAL_ENV' in os.environ

if is_venv_activated():
    print("El entorno virtual está activado.")
else:
    print("El entorno virtual NO está activado.")
