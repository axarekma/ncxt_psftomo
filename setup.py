from setuptools import setup
import setuptools

__version__ = "0.0.1"


setup(
    name="ncxt_psftomo",
    version=__version__,
    author="Axel Ekman",
    author_email="Axel.Ekman@iki.fi",
    url="https://github.com/axarekma/ncxt_psftomo",
    description="PSF projectors for SXT",
    long_description="NA",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.13.3", "matplotlib>=3.0.3", "scipy>=1.3.0","numba"],
    zip_safe=False,
)
