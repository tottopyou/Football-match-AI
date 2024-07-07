# Football Match Monitoring System
## Overview

This project focuses on monitoring football matches using a custom-trained YOLOv5x model. The system is capable of detecting and tracking players, referees, and the ball. It divides players into teams, improves ball tracking even when the ball is not visible in the current frame by leveraging information from previous frames, and identifies the player controlling the ball. Additionally, it tracks every player's speed and distance covered, displaying ball control percentages on the screen.

## Features

**Player and Referee Detection:** Identify and classify players and referees in the frame.

**Team Division:** Automatically divide players into their respective teams.

**Enhanced Ball Tracking:** Maintain ball tracking using information from previous frames when the ball is occluded.

**Ball Control Recognition:** Detect which player is controlling the ball.

**Player Statistics:** Track player speed and distance covered.

**Ball Control Display:** Show the percentage of ball control on the screen.

## Result

![output_video.mp4](https://github.com/tottopyou/Football-match-AI/assets/110258834/5f87e5a0-b0c9-410c-8d8c-6ace87a52730)

## Requirements

+ Python 3.x
+ ultralytics
+ supervision
+ OpenCV
+ NumPy
+ Matplotlib
+ Pandas
