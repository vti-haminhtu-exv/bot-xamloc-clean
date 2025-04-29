# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="card_recognition_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'flask>=2.2.0',
        'matplotlib>=3.7.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered assistant for Xâm Lốc Solo card game",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/card_recognition_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)