# Microscopic Crowd Model Using Cellular Automata

Analysis and implementation of a microscopic pedestrian crowd model based on cellular automata. Seminar paper, Faculty of Electrical Engineering and Computing (FER), University of Zagreb, 2026.

## Overview
The simulation showcases microscopic pedestrian crowd movement using cellular automata. A narrow hallway is presented and agents in the form of cells gradually make their way to the end, getting in each other's way.
This behaviour is shown to demonstrate crowd movement and avoid catastrophic events like crowd crushing. A basic round barrier is added to show potential improvements to the flow.
## Features
- grid-based pedestrian movement
- configurable hallway size, number of agents, barrier size and distance from hallway
- visualization of evacuation / flow

## Tech stack
Python, numpy, matplotlib, mesa

## Documentation
- [Seminar paper (PDF, Croatian)](docs/seminar_paper.pdf)
- [Presentation (PDF)](docs/presentation.pdf)

## Results
Simulations showed that agent behavior, corridor flow, and crowd-crush risks depend on model parameters and space geometry. Larger obstacles improved crowd distribution and reduced critical events, especially in high-crisis situations. Cellular automata proved efficient for crowd modeling, though more advanced models can simulate physical interactions more realistically.

## Author
Mislav Marinović
