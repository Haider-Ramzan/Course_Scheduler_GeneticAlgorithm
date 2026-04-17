# University Course Scheduling Optimizer using Genetic Algorithms

## Overview

This project implements a Genetic Algorithm (GA) to optimize weekly course schedules for 5 Computer Science students. The optimizer satisfies hard constraints (credits, time conflicts, course limits, availability) while optimizing soft objectives (time preferences, gap minimization, friend satisfaction, workload balance, lunch breaks). 

**Key Assumption regarding Credit Hours:**
> All courses strictly require 3 credit hours. Since all timetable slots are 1-hour blocks, courses that meet on MWF (Monday/Wednesday/Friday) naturally fulfill this with 3 classes per week (3 total hours). However, courses scheduled on Tuesday/Thursday meet only twice a week. Therefore, we assume that each 1-hour class session on a Tuesday or Thursday carries a weight of **1.5 credit hours**, correctly fulfilling the 3-credit course requirement over just two days.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Navigate to the project directory
cd "assignment 2/q2"

# 2. Install minimal dependencies
pip install matplotlib numpy
```

### Dependencies
- `matplotlib` - Plotting and timetable visualization
- `numpy` - Fast numerical operations

## Usage

### Run Full Experiment (10 runs)
```bash
python main.py
```

## Project Structure

```
q2/
├── main.py              # Main GA loop & experiment orchestration
├── chromosome.py        # Chromosome encoding/decoding & population init
├── fitness.py           # Multi-objective fitness computation & weights
├── operators.py         # Crossover (3) & mutation (4) operators + repair
├── selection.py         # Tournament + roulette wheel selection + elitism
├── utils.py             # Data loading, pre-requisite filtering, helpers
├── visualization.py     # Plotting algorithms & individual schedule rendering
├── course_catalog.json          # Input data: Course sections & times
├── student_requirements.json    # Input data: Student info & preferences
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── plots/               # Generated plots (auto-created)
│   ├── convergence.png
│   ├── diversity.png
│   ├── best_schedule.png        # Has 5 individual color-coded subplots!
│   ├── operator_analysis.png
│   └── mutation_rates.png
└── output/              # Best schedules (auto-created)
    ├── best_schedule_runN.json
    └── experiment_summary.json
```

## Architecture & Design Decisions

1. **Prerequisite Pre-Filtering:** Prerequisites are resolved entirely before the GA starts. At initialization, each student's transcript is evaluated once, and their selectable "elective pool" is securely truncated to only courses they have passed the prerequisites for. The GA operates blindly of prerequisites because its course selection space is pre-filtered, massively accelerating performance.
2. **Hardcoded Configurations:** There is no reliance on slow YAML parsed config files; all hyper-parameters (such as mutation probabilities, fitness weights, and selection constraints) have been converted to native Python constants placed directly at the top of their logically related `.py` module (e.g. fitness weights exist exclusively at the top of `fitness.py`).
3. **No External Wrappers:** Tabulate printing and Python's logging library were stripped out in favor of fast native string formatting/console printing.

## Chromosome Encoding

A chromosome is a dictionary mapping each student to their distinct course-section assignments:

```python
chromosome = {
    'S1': [('DS', 1), ('ALG', 1), ('OS', 2), ('DB', 2), ('MAD', 1)],
    'S2': [('CN', 3), ('TC', 2), ('ML', 3), ('GD', 2), ('MAD', 1)],
    ...
}
```

**Why this encoding:**
- **Compact**: Only stores `(course_id, section_id)` — day/time is looked up on demand.
- **Natural crossover**: Swapping student schedules or individual courses between parents is extremely straightforward.
- **Easy repair**: Changing a `section_id` to fix conflicts requires minimal modification while iterating over the list.

## GA Components

### Population Initialization (3 strategies)
1. **Random Valid (40%)**: Random section assignments satisfying hard constraints.
2. **Greedy Time-Based (40%)**: Prioritizes sections matching preferred student time slots.
3. **Greedy Friend-Based (20%)**: Prioritizes sections shared seamlessly with declared friends.

### Fitness Function
```
Fitness = 0.30 × Time_Preference + 0.25 × Gap_Minimization 
        + 0.20 × Friend_Satisfaction + 0.15 × Workload_Balance 
        + 0.10 × Lunch_Break - Hard_Penalties (Time Conflicts, Overload)
```

### Selection Methods
- **Tournament Selection (70%)**: Size = 5
- **Roulette Wheel (30%)**: With fitness shifting for negative values
- **Elitism**: Top 10% strictly preserved into the next generation.

### Crossover Operators (probability: 0.80)
1. **Single-Point** (35%): Cut at student boundary.
2. **Uniform** (35%): Random parent copy per student.
3. **Course-Based** (30%): Random parent copy per course.

### Mutation Operators (adaptive rates)
1. **Section Change** (0.12): Change course section.
2. **Course Swap** (0.10): Reassign two courses' sections.
3. **Time Slot Shift** (0.08): Move to different section explicitly avoiding clashing.
4. **Friend Alignment** (0.15): Directly aligns sections with friends to boost fitness.

*(Adaptive: If population diversity falls below 30%, all mutation rates boost by 1.5x).*

### Termination Criteria
Stops when any condition is met:
1. Maximum 300 generations reached.
2. Fitness improvement stalled (< 0.1%) for 40 consecutive generations.
3. Over 85% of the population suffers from convergence / high severity similarity.

## Output

After running, directly check the following analytics:
- `plots/best_schedule.png` — Creates individual schedule subgraphs (1 per student, uniquely color-coded) so distinct timetables are exceptionally clear and free of overlaps.
- `plots/convergence.png` — Best/average fitness mapped sequentially over all generations.
- `plots/diversity.png` — Visualizes GA population diversity over generations.
- `plots/operator_analysis.png` — Visual operator selection usage efficiency.
- `plots/mutation_rates.png` — Tracks real-time adaptive mutation rate tweaks.
- `output/best_schedule_best.json` — The top optimal schedule structure printed in pure JSON.
- `output/experiment_summary.json` — High-level telemetry statistics parsed across all 10 experiment runs.
