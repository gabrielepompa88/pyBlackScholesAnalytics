import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-Black-Scholes-Analytics-gabrielepompa88", # Replace with your own username
    version="0.0.1",
    author="Gabriele Pompa",
    author_email="gabriele.pompa@gmail.com",
    description="Options and Option Strategies analytics under the Black-Scholes Model for educational purpose",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielepompa88/pyBlackScholesAnalytics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)