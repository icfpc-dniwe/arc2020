#!/bin/sh

set -e

mkdir -p dist
rm -f dist/arc2020.pyz
( cd src; zip -9 -R ../dist/arc2020.pyz $(find arc2020 -name \*.py) )

archive="$(base64 -w 0 dist/arc2020.pyz)"
sed "s,REPLACE_WITH_ARCHIVE,$archive," runner.py > dist/arc2020_runner.py
