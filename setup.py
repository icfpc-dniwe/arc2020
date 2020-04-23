#!/usr/bin/env python3

from setuptools import setup


setup(
    name="arc2020",
    description="",
    version="1.0",
    package_dir={"": "src"},
    zip_safe=True,
    packages=["arc2020"],
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "tqdm",
        "torch"
    ],
    setup_requires=[],
    entry_points={
        "console_scripts": [
            "arc2020=arc2020:main"
        ]
    },
)
