from setuptools import setup, find_packages

setup(
    name="format_tree",
    version="0.1.1",
    description="A utility to plot decision trees with formatted node information.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kathy G",
    author_email="kguo715@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    python_requires=">=3.6",
    url="https://pypi.org/project/format_tree/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
