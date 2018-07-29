from setuptools import setup

setup(name='discoverSTEM',
      author = 'Andy Lamperski',
      description = 'Tutorial on coding for robot control',
      url = 'https://github.com/AndyLamperski/DiscoverSTEM',
      author_email = 'alampers@umn.edu',
      license = 'MIT',
      packages = ['discoverSTEM'],
      install_requires = [
          'jupyter',
          'numpy',
          'scipy',
          'pyglet',
          'matplotlib',
          'sympy'
      ]
      )
      

