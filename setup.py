from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stock_market_prediction',
    version='0.1.0',
    author='Andres Quast, Will Akins, Alberto Alvarez',
    author_email='andresquast@gmail.com',
    description='A machine learning project for stock market prediction using price data and sentiment analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/willakins/ML_StockPredictor_Fall2024',
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.8',
    install_requires=required,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
            'isort>=5.0.0',
            'mypy>=0.9.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'collect-data=src.data.data_collection:main',
        ],
    },
    package_data={
        'stock_market_prediction': [
            'config/*.yaml',
        ],
    },
    include_package_data=True,
)