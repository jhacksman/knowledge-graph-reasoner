from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="knowledge-graph-reasoner",
    version="0.1.0",
    author="Jack Hacksman",
    description="A self-organizing knowledge network implementation using Venice.ai and Milvus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhacksman/knowledge-graph-reasoner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
)
