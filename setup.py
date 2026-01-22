from setuptools import setup, find_packages

setup(
    name="jarvisx",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["jarvisx=jarvisx.cli:main"]},
)
