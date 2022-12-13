#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Raúl Fernández Díaz",
    author_email='Raul.FernandezDi@autonoma.cat',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description = "Computational tool for the prediction of metal-binding sites in proteins using deep convolutional neural networks.",
    entry_points={
        'console_scripts': [
            'biobrigit=biobrigit.__main__:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    package_data={'': ['*.ckpt', '*.json']},
    keywords='biobrigit',
    name='biobrigit',
    packages=find_packages(include=['biobrigit', 'biobrigit.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/RaulFD-creator/biobrigit',
    version='0.1.0',
    zip_safe=False,
)
