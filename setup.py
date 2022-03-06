
#!/usr/bin/env python

import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

setup(
    name="kaggle-nlp-disaster-tweets",
    version="0.1",
    author="Araik Tamazian",
    description="Predict which tweets are about real disasters and which one aren’t.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atamazian/fluc-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/atamazian/kaggle_nlp_disaster_tweets/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="kaggle_nlp_disaster_tweets"),
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=install_requires
)
