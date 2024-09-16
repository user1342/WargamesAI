from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="WargamesAI",  # Name of the tool/package
    version="0.1.3",  # Current version
    url="https://github.com/user1342/WargamesAI",
    packages=find_packages(),  # Automatically find package directories
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0",
        "Operating System :: Ubuntu",
    ],
    install_requires=requirements,  # Dependencies from requirements.txt
)
