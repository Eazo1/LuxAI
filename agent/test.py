# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:55:38 2025

@author: herbi
"""

import subprocess

# Run the Lux AI S3 command
command = ["luxai-s3", "agent/main.py", "agent/main.py", "--output=replay.html"]
subprocess.run(command)