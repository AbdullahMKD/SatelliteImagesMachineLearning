from setuptools import find_packages, setup

setup(
    name='LandUseClassification',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'Pillow',
        'scikit-learn',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'run_landuse_classifier=run_app:main',
        ],
    },
    author='Abdullah Durrani',
    author_email='abd15@aber.ac.uk',
    description='A GUI application for satellite image classification using K-means clustering.',
    keywords='image processing clustering k-means gui application machine learning satellite image classification',
)