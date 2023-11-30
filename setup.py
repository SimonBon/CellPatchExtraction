from setuptools import setup, find_packages

package_name = "CellPatchExtraction"

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

def get_version(package):
    with open(f"./{package}/__init__.py") as file:
        return [line.split("=")[-1].strip().strip("'") for line in file.readlines() if "__version__" in line][0]
    
print(get_version(package_name))

setup(
    name=package_name,
    version=get_version(package_name),
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Your Name",
    include_package_data=True,
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/SimonBon/FISHcreation",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
