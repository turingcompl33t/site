+++
title = 'Aided Target Recognition from Mobile Cooperative and Autonomous Sensors'
date = 2022-01-01
until = '2023'
description = 'Led the computer vision effort for ATR-MCAS, a joint effort with Carnegie Mellon’s National Robotics Engineering Center to conduct autonomous zone reconnaissance with unmanned ground and air systems.'
org = 'Artificial Intelligence Integration Center'

[[links]]
  label = 'coverage'
  url = 'https://www.militaryaerospace.com/computers/article/14167236/battlefield-artificial-intelligence-ai-target-data'
+++

I worked on ATR-MCAS during my first year at the Artificial Intelligence Integration Center, in partnership with the [National Robotics Engineering Center](https://www.nrec.ri.cmu.edu/).

The system conducts an autonomous zone reconnaissance mission in arbitrary terrain. It combines autonomous unmanned ground and air systems with object detection algorithms to identify and locate entities of interest, such as adversary vehicles, across large areas.

I led the project's computer vision effort: building, training, and evaluating our object detection models, and the data engineering needed to prepare captured data for that training and evaluation. To support it, I built a collection of tools, primarily in Python, to ingest and store data, manage datasets, annotate them with quality assurance, construct machine learning data splits, and train, evaluate, and export models. The whole pipeline was automated and orchestrated through Docker, command-line, and web interfaces.
