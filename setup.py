from setuptools import setup, find_packages

setup(
    name='datemcgill',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Web Framework
        'django==4.2.7',
        'djangorestframework==3.14.0',

        # Database
        'pymongo==4.6.0',

        # ML and Data Processing
        'torch==2.1.0',
        'numpy==1.24.3',
        'scikit-learn==1.3.2',

        # Vector Database
        'weaviate-client==3.24.1',

        # Utilities
        'python-dotenv==1.0.0',
        'typing-extensions==4.8.0',
    ],
    author='William',
    author_email='williamk.lafond@gmail.com',
    description='A McGill AI-based Dating Website',
    url='https://github.com/McGill-AI-Lab/datemcgill.git',  # Replace with your GitHub URL
)
