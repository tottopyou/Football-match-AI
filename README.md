# Football Match Monitoring System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ultralytics](https://img.shields.io/badge/ultralytics-0.1%2B-orange)
![supervision](https://img.shields.io/badge/supervision-0.1%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-yellow)
![Pandas](https://img.shields.io/badge/Pandas-1.1%2B-green)

## Overview

This project focuses on monitoring football matches using a custom-trained YOLOv5x model. The system is capable of detecting and tracking players, referees, and the ball. It divides players into teams, improves ball tracking even when the ball is not visible in the current frame by leveraging information from previous frames, and identifies the player controlling the ball. Additionally, it tracks every player's speed and distance covered, displaying ball control percentages on the screen.

## Features

**Player and Referee Detection:** Identify and classify players and referees in the frame.

**Team Division:** Automatically divides players into the appropriate teams using the color of the shirts.

**Enhanced Ball Tracking:** Maintain ball tracking using information from previous frames when the ball is occluded.

**Ball Control Recognition:** Detect which player is controlling the ball.

**Player Statistics:** Track player speed and distance covered.

**Ball Control Display:** Show the percentage of ball control on the screen.

## Result

![image](https://github.com/tottopyou/Football-match-AI/assets/110258834/9c936191-b5d0-4d67-88a5-5a7f499bdfe2)

### Full video you can see here :  [Full Video](https://github.com/tottopyou/Football-match-AI/assets/110258834/cbfa0e13-d613-4874-9337-32507d63ea1b)


## Requirements

+ Python 3.x
+ ultralytics
+ supervision
+ OpenCV
+ NumPy
+ Matplotlib
+ Pandas
