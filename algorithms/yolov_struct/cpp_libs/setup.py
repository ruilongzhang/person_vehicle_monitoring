from distutils.core import setup

setup (name = 'nms',
       version = '1.0',
       description = """Install precompiled DeepStream Python bindings for tracker metadata extension""",
       packages=[''],
       package_data={'': ['cpu_nms.cpython-36m-x86_64-linux-gnu.so']},
       )
