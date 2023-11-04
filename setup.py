from setuptools import setup, find_packages

requirements = [
    "tqdm",
]

setup(
    name="wetts",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wetts = wetts.cli.tts:main",
    ]},
)
