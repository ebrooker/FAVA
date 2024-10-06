#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Ezra Brooker",
    author_email='ebrooker@fsu.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Fluid Advanced Variable Analysis: package for performing advanced analysis calculations on computational fluid dynamics data.",
    entry_points={
        'console_scripts': [
            'fava=fava.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fava',
    name='fava',
    packages=find_packages(include=['fava', 'fava.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ebrooker/fava',
    version='0.0.0.alpha',
    zip_safe=False,
)