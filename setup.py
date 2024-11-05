from setuptools import setup, find_packages

setup(
    name="keras_reservoir_computing",
    version="0.1.0",
    description="A package for training and using Echo State Networks and Reservoir Computing in general.",
    author="Daniel Estevez",
    author_email="kemossabee@gmail.com",
    url="https://github.com/ZentropyUH/ESN",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "tensorflow==2.17.0",
        "rich==13.8.1",
        "networkx==3.4.2",
        "scipy==1.14.1",
        "keras==3.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
