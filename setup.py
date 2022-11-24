from setuptools import setup, find_packages
from os import path

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Inteded Audience :: Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: GPL-3.0 License',
    'Programming Language :: Python :: 3'
]

if __name__ == "__main__":
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            install_requires.append(req)
            
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='EEGRAPH',
    version='0.1.14',
    description='',
    url='https://github.com/ufvceiec/EEGRAPH',
    author='CEIEC',
    license= 'GPL-3.0',
    classifers=classifiers,
    keywords='',
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown'
)

