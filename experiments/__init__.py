"""
Experiment scripts.

Each script is standalone and is run as ``python experiments/<script>.py``.
They are launched in separate processes by ``run_all.py`` so that the
experiments which rely on the global NumPy generator cannot influence one
another.
"""
