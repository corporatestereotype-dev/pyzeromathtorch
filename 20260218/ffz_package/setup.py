from setuptools import setup, find_packages

setup(
    name='ffz_package',
    version='1.0.0-godmode',
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    author='Corporate',
    author_email='corporate@example.com',
    description='Comprehensive FFZ Algebra and Extensions Package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/corporate/ffz_package',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'ffz-run=ffz_package.examples.main:main',
        ],
    },
    include_package_data=True,
)
