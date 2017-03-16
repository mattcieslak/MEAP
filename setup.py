#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
      name="MEAP",
      version="1.1.0",
      description="A program for analyzing acqknowledge files",
      author="Matt Cieslak and Will Ryan",
      author_email="matthew.cieslak@psych.ucsb.edu",
      install_requires=[
          "ez_setup",
          "joblib",
          "scikit-learn==0.17.1",
          "scikit-image",
          "traits",
          "traitsui",
          #"pyqt<5",
          "kiwisolver",
          "nibabel",
          "bioread==0.9.5",
          "xlrd",
          #"chaco",
          "numpy",
          "scipy",
          "pandas"],
      packages=find_packages(),
      package_data={
          "":[
              "resources/logo48x38.png",
              "resources/logo512x512.png"
            ]
          },
      entry_points= {
          "gui_scripts": [
          "meap_analyze = meap.easy_launch:analyze",
          "meap_preprocess = meap.easy_launch:preprocess" ]
          }
          
     )
