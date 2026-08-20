#!/usr/bin/env python

import os
import logging
import subprocess

def module_list() -> str:
    # LOADEDMODULES may not be set if no modules are loaded
    modules = os.environ.get('LOADEDMODULES', '')
    loaded = ', '.join(m for m in modules.split(os.pathsep) if m)
    logging.info('MODULES: ' + loaded)
    return loaded

def module_purge() -> None:
    # exec() is intentional: modulecmd returns Python statements
    # that modify os.environ in the calling process
    result = subprocess.run(
        ['modulecmd', 'python', 'purge'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.warning("module purge failed: %s", result.stderr.decode())
        return
    exec(result.stdout.decode())

def module_unload(module: list = []) -> None:
    result = subprocess.run(
        ['modulecmd', 'python', 'unload'] + module,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.warning("module unload failed: %s", result.stderr.decode())
        return
    exec(result.stdout.decode())

def module_load(module: list = []) -> None:
    result = subprocess.run(
        ['modulecmd', 'python', 'load'] + module,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.warning("module load failed: %s", result.stderr.decode())
        return
    exec(result.stdout.decode())
