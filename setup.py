from setuptools import setup, find_packages

setup(
    name='idkrom',
    version='0.1.0',  # You can adjust the version number as needed
    description='A Reduced Order Modeling (ROM) framework',
    author='Ander Alvarez Sanz',  # Replace with your name
    author_email='andersz.alvarez@gmail.com',  # Replace with your email
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)