from setuptools import setup, find_packages

setup(
    name='LandUseClassification',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'Pillow',
        'scikit-learn',
        'sklearn',
        'tkinter',
        'cv2',
        'math',
    ],
    entry_points={
        'console_scripts': [
            'run_landuse_classifier=run_app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'resources': ['*.*'],
    },
    author='Abdullah Durrani',
    author_email='abd15@aber.ac.uk',
    description='A GUI application for satellite image classification using K-means clustering.',
    keywords='image processing clustering k-means gui application machine learning satellite image classification',
)
