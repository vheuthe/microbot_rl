from setuptools import setup, find_packages

def parse_requirements(filename):
    # Read the requirements from the file
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="microbot_rl",
    version="1.0",
    description="train microbot swarms",
    author="vheuthe",
    author_email="veit-lorenz.heuthe@uni.kn",
    url="https://github.com/vheuthe/microbot_rl",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)