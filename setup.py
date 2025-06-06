from setuptools import setup, find_packages
import pathlib

this_directory = pathlib.Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")
except Exception:
    long_description = "A Python package for advanced formatting, analysis, and visualization of scikit-learn decision trees."

setup(
    name="format_tree",
    version="0.1.4",
    description="Advanced formatting, analysis, and visualization utilities for scikit-learn decision trees.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kathy G",
    author_email="kguo715@gmail.com",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "numpy>=1.18",
        "matplotlib>=3.2",
        "scikit-learn>=0.24",
        "pandas>=1.0"
    ],
    python_requires=">=3.7",
    url="https://github.com/kk715/format_tree",
    project_urls={
        "Documentation": "https://github.com/kk715/format_tree#readme",
        "Source": "https://github.com/kk715/format_tree",
        "Tracker": "https://github.com/kk715/format_tree/issues",
        "PyPI": "https://pypi.org/project/format_tree/"
    },
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Framework :: Matplotlib",
    ],
    keywords="decision tree, visualization, sklearn, matplotlib, pandas, analysis, data science, machine learning",
)
