from setuptools import setup, find_packages
from os import path

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Inteded Audience :: Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: GPL-3.0 License',
    'Programming Language :: Python :: 3'
]

_DOC_PACKAGES = {'mkdocs', 'mkdocs-material', 'mkdocstrings', 'pymdown-extensions'}

if __name__ == "__main__":
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            # Skip blank lines, comments, and documentation-only packages
            if not req or req.startswith('#'):
                continue
            pkg_name = req.split('>=')[0].split('==')[0].split('[')[0].lower()
            if pkg_name in _DOC_PACKAGES:
                continue
            install_requires.append(req)

    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='EEGRAPH',
    version='0.1.17',
    description='Open-source Python library for modeling EEGs as graphs',
    url='https://github.com/albertonogales/EEGRAPH',
    author='Alberto Nogales, Álvaro José García-Tejedor',
    author_email='alberto.nogales@uah.es',
    license='GPL-3.0',
    classifiers=classifiers,
    keywords='EEG graph connectivity brain neuroscience',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=install_requires,
    extras_require={
        'test': ['pytest>=7.0', 'pytest-cov>=4.0'],
        'docs': [
            'mkdocs>=1.5',
            'mkdocs-material>=9.0',
            'mkdocstrings[python]>=0.20',
            'pymdown-extensions>=10.0',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)

