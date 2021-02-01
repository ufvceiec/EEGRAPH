from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Inteded Audience :: Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

if __name__ == "__main__":
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            install_requires.append(req)

setup(
    name='eegraph',
    version='0.0.20',
    description='',
    url='',
    author='CEIEC',
    license= '',
    classifers=classifiers,
    keywords='',
    packages=find_packages(),
    install_requires=install_requires
)

