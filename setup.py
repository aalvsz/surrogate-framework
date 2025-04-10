from setuptools import setup, find_packages

setup(
    name='idkROM',
    version='0.1.0',  # You can adjust the version number as needed
    description='A Reduced Order Modeling (ROM) framework',
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    packages=find_packages(where='idkROM'),  # Tells setuptools to find packages within the 'src' directory
    package_dir={'': 'idkROM'},  # Maps the root package to the 'src' directory
    py_modules=['main'],  # If main.py is meant to be a top-level module
    entry_points={
        'console_scripts': [
            'idkrom=main:main',  # Creates a command 'idkrom' that runs the 'main' function from main.py
        ],
    },
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
        'plotly>=6.0.0',  # Specify a version if needed
        'joblib',
        'plotly-express>=0.4.1',  # Use the correct package name and specify a version if needed
        # 'plotly.io',  # Remove this line as it's part of the 'plotly' package
        'pymoo',
        'scipy',
        # Add any other dependencies your project has
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            # Add other development dependencies
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',  # Adjust as needed (Alpha, Beta, Stable, etc.)
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',  # Choose the appropriate license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)