from setuptools import find_packages, setup

setup(
    name="smc",
    version="0.0.1",
    author="Your name (or your organization/company/team)",
    author_email="alexjameschan@gmail.com",
    description="A short description of the project.",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="smc.tests.test_all.suite",
)
