from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    description="A Package for Creating AI",
    author="Olivier",
    author_email="luowensheng2018@gmail.com",
    entry_points={
        "console_scripts": [
            "ai=ai.cli:main"
        ]
    }
)
