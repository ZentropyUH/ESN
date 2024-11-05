from setuptools import setup, find_packages

setup(
    name="keras_reservoir_computing",
    version="0.1.0",
    description="A package for training and using Echo State Networks and Reservoir Computing in general.",
    author="Daniel Estevez",
    author_email="kemossabee@gmail.com",
    url="https://github.com/ZentropyUH/ESN",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "tensorflow",
        "rich",
        "networkx",
        "scipy",
        "keras",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
