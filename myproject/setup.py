from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["pandas==1.3.4"]

setup(
    name="trainers",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="basic regression from tabular data.",
)
