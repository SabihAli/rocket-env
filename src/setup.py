from setuptools import setup
from.rocket_env import __version__

setup(
    name='rocket_sim_6dof',
    version=__version__,
    install_requires=["gym==0.21", 'scipy==1.7.3','pyvista',],
    )
