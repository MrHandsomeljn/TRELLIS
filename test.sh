#!/bin/sh
cd $(dirname $0)
source .venv/bin/activate
python test.py $1 --skip_glb --skip_nerf