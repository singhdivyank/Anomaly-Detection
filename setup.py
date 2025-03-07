from setuptools import setup, find_packages

setup(
    name='cs6140_ml_project',
    version='1.0.0',
    description='Analyzing Household Responses to Dynamic Time-of-Use Pricing Signals',
    author='Ishan Biswas, Divyank Singh, Sri Sai Teja Mettu Srinivas',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "joblib",
        "tensorflow",
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts if needed, e.g.:
            # "cs6140_ml_project=src.main:main",
        ],
    },
)
