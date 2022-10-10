from setuptools import find_packages, setup

setup(
    name="smc",
    version="0.0.1",
    author="Alex J. Chan",
    author_email="ajc340@cam.ac.uk",
    description="Synthetic Model Combination.",
    url="https://github.com/XanderJC/synthetic-model-combination",
    packages=find_packages(),
    test_suite="smc.tests.test_all.suite",
)
