"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py install
"""
import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    # parse_requirements() returns generator of pip.req.InstallRequirement objects
    # (This API is deprecated, cf. https://stackoverflow.com/a/59971236)
    if pip_version >= [20, 0]:
        from pip._internal.req import parse_requirements
        from pip._internal.network.session import PipSession
        req = 'requirement'
    elif pip_version >= [10, 0]:
        from pip._internal.req import parse_requirements
        from pip.download import PipSession
        req = 'req'
    elif pip_version >= [6, 0]:
        from pip.req import parse_requirements
        from pip.download import PipSession
        req = 'req'
    else:
        from pip.req import parse_requirements
        req = 'req'
    raw = parse_requirements(file_path, session=PipSession())
    return [str(getattr(i, req)) for i in raw]

# may raise exceptions if requirements cannot be parsed
# - let them through
install_reqs = _parse_requirements("requirements.txt")

setup(
    name='mask-rcnn',
    version='2.1',
    url='https://github.com/matterport/Mask_RCNN',
    author='Matterport',
    author_email='waleed.abdulla@gmail.com',
    license='MIT',
    description='Mask R-CNN for object detection and instance segmentation',
    packages=["mrcnn"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.4',
    long_description="""This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. 
The model generates bounding boxes and segmentation masks for each instance of an object in the image. 
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.""",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="image instance segmentation object detection mask rcnn r-cnn tensorflow keras",
)
