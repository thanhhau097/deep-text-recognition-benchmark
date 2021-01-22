from setuptools import setup
from io import open

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')


def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='lionelocr',
    packages=['lionelocr'],
    package_dir={'lionelocr': ''},
    include_package_data=True,
    # entry_points={"console_scripts": ["paddleocr= paddleocr.paddleocr:main"]},
    version='0.0.1',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Lionel OCR',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/thanhhau097/deep-text-recognition-benchmark',
    download_url='https://github.com/thanhhau097/deep-text-recognition-benchmark.git',
    keywords=[
        'ocr'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )