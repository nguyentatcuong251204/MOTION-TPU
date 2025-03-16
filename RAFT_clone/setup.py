from setuptools import find_packages, setup

setup(
    name="raft",
    version="1.0",
    author="unknown",
    url="unknown",
    description="raft",
    keywords = [
    'artificial intelligence',
    'optical flow',
    'transformers'
    ],
    install_requires=[
        'einops>=0.3',
        'torch>=1.6'
    ],
    packages=find_packages(exclude=("configs", "tests","static")),
)