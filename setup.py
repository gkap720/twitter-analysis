from setuptools import setup, find_packages
    
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='twitter-sentiment-analysis',
    version="0.0.1",
    author='Greg Kappes',
    author_email='gkap720@gmail.com',
    url='https://github.com/gkap720/twitter-sentiment-analysis',
    description='Trains and deploys model analyzing sentiments of tweets',

    # Packages and depencies
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=required
)