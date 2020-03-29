#!/bin/sh

set -e

mkdir -p dist
python -m zipapp --output dist/arc2020.pyz --compress src/arc2020

archive="$(base64 -w 0 dist/arc2020.pyz)"
sed "s,REPLACE_WITH_ARCHIVE,$archive," runner.py > dist/arc2020_runner.py
