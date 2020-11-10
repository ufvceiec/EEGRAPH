from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Inteded Audience :: Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='to_define',
    version='0.0.1',
    description='',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='CEIEC',
    license= '',
    classifers=classifiers,
    keywords='',
    packages=find_packages(),
    install_requires=['']
)