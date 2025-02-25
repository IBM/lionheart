from setuptools import setup, find_packages
import os

cwd = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(cwd, 'lionheart', '__version__.py')
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

setup(
    name="lionheart",
    version=version,
    description="LionHeart: A Layer-based Mapping Framework for Heterogeneous Systems with Analog In-Memory Computing Tiles.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Corey Lammie and Yuxuan Wang and Flavio Ponzina and Joshua Klein and Hadjer Benmeziane and Marina Zapater and Irem Boybat and Abu Sebastian and Giovanni Ansaloni and David Atienza',
    author_email='corey.lammie@ibm.com',
    license='Apache 2.0',
    keywords=['analog in-memory computing', 'heterogeneous systems', 'lionheart'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)