from setuptools import setup, find_packages

setup(
    name='placenta_segmentation',
    version='0.1.0',
    description='Deep‐learning based segmentation of placenta and uterine cavity in prenatal MRI',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'h5py>=2.10.0',
        'tensorflow>=2.4.0',
        'opencv-python>=4.2.0.32',
        'tqdm>=4.0.0',
        'matplotlib>=3.1.0',
        'Pillow>=7.0.0'
    ],
    entry_points={
        'console_scripts': [
            'psg-train=scripts.train:main',
            'psg-predict=scripts.predict:main',
            'psg-evaluate=scripts.evaluate:main',
        ],
    },
    python_requires='>=3.7',
)
