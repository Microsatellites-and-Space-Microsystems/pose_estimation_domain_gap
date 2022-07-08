from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name="ai22_building",
    version="1.0",
    url="https://github.com/alelotti96/",
    author="Alessandro Lotti",
    author_email="alessandrolotti1996@gmail.com",
    license="Apache",
    python_requires=">=3.6.0,<3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),

    install_requires=['tensorflow>=2.7.0','tensorflow-addons','numpy','scipy','tensorflow_model_optimization','opencv-python'],

)