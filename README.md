# ITC2007 Examination Timetabling

A Python-based solver for the International Timetabling Competition 2007 (ITC2007) Examination Timetabling Problem (Track 1). This project supports both hard and soft constraints and produces optimized exam schedules using Gurobi as the optimization solver.

## Problem Description

Schedule each exam into a period and a room. Multiple exams can share the same room during the same period.

### Hard Constraints
- **Exam conflict:** Two exams that share students must not occur in the same period.
- **Room capacity:** A room’s seating capacity must suffice at all times.
- **Period duration:** A period’s duration must suffice for all of its exams.
- **Period-related constraints:**
  - Coincidence: Two specified exams must use the same period (but possibly another room).
  - Exclusion: Two specified exams must not use the same period.
  - After: A specified exam must occur in a period after another specified exam’s period.
- **Room-related constraints:**
  - Exclusive: One specified exam should not have to share its room with any other exam.

### Soft Constraints
- The same student should not have two exams in a row.
- The same student should not have two exams on the same day.
- Period spread: Two exams that share students should be a number of periods apart.
- Mixed durations: Two exams that share a room should not have different durations.
- Front load: Large exams should be scheduled earlier in the schedule.
- Period penalty: Some periods have a penalty when used.
- Room penalty: Some rooms have a penalty when used.

## Dataset

Uses data from the **ITC2007 Exam Timetabling Competition**. Place your `.exam` file (e.g., `exam_comp_set1.exam`) in the `datasets/` directory.

## Requirements

- Python 3.7+
- [Gurobi Optimizer](https://www.gurobi.com/)
- Gurobi Python package (`gurobipy`)

## Installation

1. Install Python 3.7 or higher.
2. Install Gurobi and obtain a license (free for academics): https://www.gurobi.com/academia/academic-program-and-licenses/
3. Install required Python packages:
   ```bash
   pip install gurobipy
   ```

## Usage

Run the main script to solve a dataset:

```bash
python main.py --dataset DATASET_NAME.exam [--time-limit SECONDS] [--analyze-only] [--mock] [--debug] [--enhanced]
```

- If `--dataset` is not provided, you will be prompted to select one interactively.
- `--time-limit`: Set a time limit for the solver (in seconds).
- `--analyze-only`: Only analyze the dataset without solving.
- `--mock`: Create a mock solution without using Gurobi.
- `--debug`: Enable debug output.
- `--enhanced`: Use the enhanced solver (recommended).

### Example

```bash
python main.py --dataset exam_comp_set1.exam --time-limit 300 --enhanced --debug
```

Results will be saved in the `results/` directory under a subfolder named after the dataset.

## Project Structure

- `main.py` — Main entry point for running experiments and solving instances
- `datasets/` — Contains all `.exam` dataset files
- `results/` — Output directory for solutions and statistics
- `utils/parser.py` — Dataset parsing utilities
- `utils/solver.py` — Standard solver implementation
- `utils/solver_enhanced.py` — Enhanced solver implementation

## Credits

- Based on the ITC2007 Exam Timetabling Competition datasets and rules.
- Gurobi Optimizer for solving the optimization problem.
- For more information, see the [ITC2007 Competition Page](https://www.itc2007.org/).

## License

This project is for academic and research purposes. Please check dataset and solver licenses for further details.
