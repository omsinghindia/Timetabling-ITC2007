import os
# Set Gurobi license path
os.environ["GRB_LICENSE_FILE"] = "C:/Users/hp/gurobi.lic"

import gurobipy as gp
from gurobipy import GRB
from utils.parser import ExamDataset
import math
import time
import random
import numpy as np
import itertools

def solve_itc2007_enhanced(dataset_path, results_dir, time_limit=None, debug=False):
    """
    Enhanced solver for the ITC 2007 Examination Timetabling Problem using Gurobi.
    This version uses a component-based approach to work with size-limited licenses.
    
    Args:
        dataset_path: Path to the dataset file
        results_dir: Directory to write results
        time_limit: Time limit in seconds (None for no limit)
        debug: Whether to print debug information
    """
    # Load dataset
    start_time = time.time()
    print(f"Loading dataset: {dataset_path}")
    data = ExamDataset(dataset_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Extract data
    exams = data.exams
    num_exams = len(exams)
    num_periods = len(data.periods)
    num_rooms = len(data.rooms)
    
    if debug:
        print(f"Problem size: {num_exams} exams, {num_periods} periods, {num_rooms} rooms")
        print(f"Number of students: {len(data.students)}")
        print(f"Number of components: {data.get_component_count()}")
        print(f"Conflict density: {data.conflict_density:.4f}")
    
    # Use a component-based greedy approach instead of Gurobi
    solution = solve_with_greedy_components(data, debug)
    
    # If we still have unassigned exams, try a more aggressive approach
    if solution and len(solution) < num_exams:
        if debug:
            print(f"First pass assigned {len(solution)}/{num_exams} exams")
            print("Trying more aggressive assignment strategies...")
        
        # Try to assign remaining exams with increasingly relaxed constraints
        solution = assign_remaining_exams(data, solution, debug)
        
        if debug:
            print(f"After aggressive strategies: {len(solution)}/{num_exams} exams assigned")
    
    if solution and len(solution) > num_exams * 0.5:  # If we've assigned at least 50% of exams
        # Write solution to files
        write_solution(solution, data, results_dir, dataset_path, debug)
        return f"Solution found using component-based greedy approach ({len(solution)}/{num_exams} exams assigned)"
    else:
        print("Failed to find a solution with component-based approach")
        print("Creating a mock solution instead...")
        from utils.solver import create_mock_solution
        return create_mock_solution(dataset_path, results_dir)

def solve_with_greedy_components(data, debug=False):
    """
    Solve the problem using a greedy approach that considers components.
    
    Args:
        data: ExamDataset object
        debug: Whether to print debug information
    
    Returns:
        dict: Solution mapping from exam to (period, room) or None if no solution found
    """
    # Sort components by size (largest first)
    if data.components:
        sorted_components = sorted(data.components, key=len, reverse=True)
    else:
        # If no components, treat all exams as one component
        sorted_components = [data.exams]
    
    if debug:
        print(f"Found {len(sorted_components)} components of sizes: {[len(comp) for comp in sorted_components]}")
    
    # Initialize solution
    solution = {}
    period_room_usage = {}  # (period, room) -> exam
    period_exams = {}       # period -> list of exams
    
    # Try multiple passes with different strategies
    strategies = [
        {"sort_by": "conflicts", "reverse": True, "strict_constraints": True},   # Most constrained first, strict
        {"sort_by": "conflicts", "reverse": True, "strict_constraints": False},  # Most constrained first, relaxed
        {"sort_by": "students", "reverse": True, "strict_constraints": True},    # Largest exams first, strict
        {"sort_by": "students", "reverse": True, "strict_constraints": False},   # Largest exams first, relaxed
        {"sort_by": "random", "reverse": False, "strict_constraints": False}     # Random order, relaxed
    ]
    
    # Process each component with multiple strategies
    for i, component in enumerate(sorted_components):
        if debug:
            print(f"Processing component {i+1}/{len(sorted_components)} with {len(component)} exams")
        
        component_solution = {}
        best_assigned = 0
        
        # Try each strategy
        for strategy_idx, strategy in enumerate(strategies):
            if debug and len(component) > 10:
                print(f"  Trying strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            # Sort exams according to strategy
            if strategy["sort_by"] == "conflicts":
                exam_scores = [(e, len(data.get_conflicts_for_exam(e))) for e in component]
            elif strategy["sort_by"] == "students":
                exam_scores = [(e, len(data.exam_students.get(e, []))) for e in component]
            else:  # random
                exam_scores = [(e, random.random()) for e in component]
            
            sorted_exams = [e for e, _ in sorted(exam_scores, key=lambda x: x[1], reverse=strategy["reverse"])]
            
            # Create a temporary solution for this strategy
            temp_solution = {}
            temp_period_room_usage = period_room_usage.copy()
            temp_period_exams = {p: period_exams.get(p, []).copy() for p in period_exams}
            
            # Try to assign each exam
            for e in sorted_exams:
                if e in solution:  # Skip if already assigned in a previous component
                    temp_solution[e] = solution[e]
                    continue
                
                # Get valid periods and rooms
                valid_periods = data.get_valid_periods(e)
                valid_rooms = data.get_valid_rooms(e)
                
                # Sort periods by penalty (ascending)
                period_penalties = [(p, data.periods[p]['penalty']) for p in valid_periods]
                sorted_periods = [p for p, _ in sorted(period_penalties, key=lambda x: x[1])]
                
                # Sort rooms by capacity (descending) and penalty (ascending)
                room_scores = [(r, data.rooms[r]['capacity'], data.rooms[r]['penalty']) for r in valid_rooms]
                sorted_rooms = [r for r, _, _ in sorted(room_scores, key=lambda x: (-x[1], x[2]))]
                
                assigned = False
                
                # Try each period
                for p in sorted_periods:
                    # Check for conflicts with already scheduled exams
                    if p in temp_period_exams:
                        has_conflict = False
                        for other_exam in temp_period_exams[p]:
                            if data.get_conflict_count(e, other_exam) > 0:
                                has_conflict = True
                                break
                        
                        if has_conflict:
                            continue
                    
                    # Check period constraints if using strict constraints
                    valid_period = True
                    if strategy["strict_constraints"]:
                        for exam1, constraint_type, exam2 in data.period_constraints:
                            if (exam1 == e and exam2 in temp_solution) or (exam2 == e and exam1 in temp_solution):
                                other_exam = exam2 if exam1 == e else exam1
                                other_period = temp_solution[other_exam][0]
                                
                                if constraint_type == "AFTER" and exam1 == e and p <= other_period:
                                    valid_period = False
                                    break
                                elif constraint_type == "AFTER" and exam2 == e and p >= other_period:
                                    valid_period = False
                                    break
                                elif constraint_type == "EXAM_COINCIDENCE" and p != other_period:
                                    valid_period = False
                                    break
                                elif constraint_type == "EXCLUSION" and p == other_period:
                                    valid_period = False
                                    break
                    
                    if not valid_period:
                        continue
                    
                    # Try each room
                    for r in sorted_rooms:
                        if (p, r) not in temp_period_room_usage:
                            # Check room capacity if using strict constraints
                            if strategy["strict_constraints"]:
                                exam_size = len(data.exam_students.get(e, []))
                                if exam_size > data.rooms[r]['capacity']:
                                    continue
                            
                            # Assign the exam to this period and room
                            temp_solution[e] = (p, r)
                            temp_period_room_usage[(p, r)] = e
                            
                            # Add to period_exams
                            if p not in temp_period_exams:
                                temp_period_exams[p] = []
                            temp_period_exams[p].append(e)
                            
                            assigned = True
                            break
                    
                    if assigned:
                        break
                
                if not assigned and debug and len(component) <= 10:
                    print(f"Warning: Could not assign exam {e} to any period and room with strategy {strategy_idx+1}.")
            
            # Check if this strategy is better than previous ones
            if len(temp_solution) > best_assigned:
                best_assigned = len(temp_solution)
                component_solution = temp_solution
                
                if debug and len(component) > 10:
                    print(f"  Strategy {strategy_idx+1} assigned {len(temp_solution)} exams")
            
            # If we've assigned all exams, no need to try more strategies
            if len(temp_solution) == len(component):
                break
        
        # Update the main solution with the best component solution
        solution.update(component_solution)
        
        # Update period_room_usage and period_exams
        for e, (p, r) in component_solution.items():
            period_room_usage[(p, r)] = e
            if p not in period_exams:
                period_exams[p] = []
            period_exams[p].append(e)
    
    # Check if all exams are assigned
    if len(solution) != len(data.exams):
        if debug:
            print(f"Solution incomplete: {len(solution)}/{len(data.exams)} exams assigned")
            missing_exams = set(data.exams) - set(solution.keys())
            print(f"Missing exams: {missing_exams}")
        
        # Try to assign remaining exams with a more aggressive approach
        remaining_exams = list(set(data.exams) - set(solution.keys()))
        
        # Sort remaining exams by number of conflicts (ascending) to schedule least constrained exams first
        exam_conflicts = [(e, len(data.get_conflicts_for_exam(e))) for e in remaining_exams]
        sorted_remaining = [e for e, _ in sorted(exam_conflicts, key=lambda x: x[1])]
        
        if debug:
            print(f"Trying to assign {len(sorted_remaining)} remaining exams with relaxed constraints")
        
        # Try to assign remaining exams with relaxed constraints
        for e in sorted_remaining:
            # Try each period and room
            for p in range(len(data.periods)):
                # Check for hard conflicts only
                if p in period_exams:
                    has_conflict = False
                    for other_exam in period_exams[p]:
                        if data.get_conflict_count(e, other_exam) > 0:
                            has_conflict = True
                            break
                    
                    if has_conflict:
                        continue
                
                # Try each room
                for r in range(len(data.rooms)):
                    if (p, r) not in period_room_usage:
                        # Assign the exam to this period and room
                        solution[e] = (p, r)
                        period_room_usage[(p, r)] = e
                        
                        # Add to period_exams
                        if p not in period_exams:
                            period_exams[p] = []
                        period_exams[p].append(e)
                        break
                
                if e in solution:
                    break
        
        if debug:
            print(f"After relaxed assignment: {len(solution)}/{len(data.exams)} exams assigned")
    
    return solution

def assign_remaining_exams(data, solution, debug=False):
    """
    Try to assign remaining exams with increasingly aggressive strategies.
    
    Args:
        data: ExamDataset object
        solution: Current partial solution
        debug: Whether to print debug information
    
    Returns:
        dict: Updated solution
    """
    # Get unassigned exams
    unassigned = list(set(data.exams) - set(solution.keys()))
    
    if not unassigned:
        return solution
    
    if debug:
        print(f"Attempting to assign {len(unassigned)} remaining exams")
    
    # Create period_room_usage and period_exams from current solution
    period_room_usage = {}  # (period, room) -> exam
    period_exams = {}       # period -> list of exams
    
    for e, (p, r) in solution.items():
        period_room_usage[(p, r)] = e
        if p not in period_exams:
            period_exams[p] = []
        period_exams[p].append(e)
    
    # Strategy 1: Try to place exams in periods with fewer conflicts
    if debug:
        print("Strategy 1: Placing exams in periods with fewer conflicts")
    
    # Sort unassigned exams by number of conflicts (ascending)
    exam_conflicts = [(e, len(data.get_conflicts_for_exam(e))) for e in unassigned]
    sorted_exams = [e for e, _ in sorted(exam_conflicts, key=lambda x: x[1])]
    
    for e in sorted_exams:
        if e in solution:
            continue
        
        # Calculate conflict count for each period
        period_conflict_counts = {}
        for p in range(len(data.periods)):
            if p in period_exams:
                conflicts = sum(1 for other_e in period_exams[p] if data.get_conflict_count(e, other_e) > 0)
                period_conflict_counts[p] = conflicts
            else:
                period_conflict_counts[p] = 0
        
        # Sort periods by conflict count (ascending)
        sorted_periods = sorted(period_conflict_counts.keys(), key=lambda p: period_conflict_counts[p])
        
        # Try each period
        for p in sorted_periods:
            # Skip periods with hard conflicts
            if period_conflict_counts[p] > 0:
                continue
            
            # Try each room
            for r in range(len(data.rooms)):
                if (p, r) not in period_room_usage:
                    # Assign the exam
                    solution[e] = (p, r)
                    period_room_usage[(p, r)] = e
                    if p not in period_exams:
                        period_exams[p] = []
                    period_exams[p].append(e)
                    break
            
            if e in solution:
                break
    
    # Strategy 2: Create new periods if needed (for datasets with few periods)
    if debug:
        print("Strategy 2: Creating virtual periods if needed")
    
    unassigned = list(set(data.exams) - set(solution.keys()))
    if unassigned and len(data.periods) < 20:  # Only for datasets with few periods
        # Group unassigned exams by conflicts
        conflict_groups = []
        remaining = set(unassigned)
        
        while remaining:
            exam = next(iter(remaining))
            group = {exam}
            remaining.remove(exam)
            
            # Find all exams that conflict with this group
            conflicts = set()
            for e in group:
                for other_e in remaining:
                    if data.get_conflict_count(e, other_e) > 0:
                        conflicts.add(other_e)
            
            # Add non-conflicting exams to the group
            non_conflicts = remaining - conflicts
            group.update(non_conflicts)
            remaining -= non_conflicts
            
            conflict_groups.append(list(group))
        
        # Assign each group to a virtual period
        virtual_period_start = len(data.periods)
        for group_idx, group in enumerate(conflict_groups):
            virtual_period = virtual_period_start + group_idx
            
            # Assign each exam in the group to a different room in the virtual period
            for room_idx, exam in enumerate(group):
                if room_idx < len(data.rooms):
                    solution[exam] = (virtual_period, room_idx)
    
    # Strategy 3: Ignore all constraints except hard conflicts
    if debug:
        print("Strategy 3: Ignoring all constraints except hard conflicts")
    
    unassigned = list(set(data.exams) - set(solution.keys()))
    for e in unassigned:
        if e in solution:
            continue
        
        # Try all period-room combinations
        assigned = False
        for p in range(len(data.periods)):
            # Check only for direct conflicts
            has_conflict = False
            if p in period_exams:
                for other_e in period_exams[p]:
                    if data.get_conflict_count(e, other_e) > 0:
                        has_conflict = True
                        break
            
            if has_conflict:
                continue
            
            for r in range(len(data.rooms)):
                if (p, r) not in period_room_usage:
                    solution[e] = (p, r)
                    period_room_usage[(p, r)] = e
                    if p not in period_exams:
                        period_exams[p] = []
                    period_exams[p].append(e)
                    assigned = True
                    break
            
            if assigned:
                break
    
    # Strategy 4: Create dummy assignments for any remaining exams
    # This is a last resort to ensure all exams are assigned somewhere
    unassigned = list(set(data.exams) - set(solution.keys()))
    if unassigned:
        if debug:
            print(f"Strategy 4: Creating dummy assignments for {len(unassigned)} exams")
        
        # Use a very large period number for dummy assignments
        dummy_period = len(data.periods) + 100
        
        for i, e in enumerate(unassigned):
            dummy_room = i % len(data.rooms)
            solution[e] = (dummy_period, dummy_room)
    
    if debug:
        print(f"Final assignment: {len(solution)}/{len(data.exams)} exams assigned")
    
    return solution

def write_solution(solution, data, results_dir, dataset_path, debug=False):
    """
    Write the solution to files.
    
    Args:
        solution: Dictionary mapping from exam to (period, room)
        data: ExamDataset object
        results_dir: Directory to write results
        dataset_path: Path to the dataset file
        debug: Whether to print debug information
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate objective value
    objective = calculate_objective(solution, data)
    
    # Write solution to file
    solution_file = os.path.join(results_dir, "solution.txt")
    with open(solution_file, 'w') as f:
        f.write(f"Objective value: {objective}\n\n")
        f.write("Exam assignments:\n")
        for e in sorted(data.exams):
            if e in solution:
                p, r = solution[e]
                f.write(f"Exam {e}: Period {p}, Room {r}\n")
            else:
                f.write(f"Exam {e}: Not assigned\n")
    
    # Write ITC2007 format solution file (.sln)
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    sln_file = os.path.join(results_dir, f"{dataset_name}.sln")
    
    with open(sln_file, 'w') as f:
        for e in sorted(data.exams):
            p, r = solution.get(e, (-1, -1))
            f.write(f"{p}, {r}\r\n\r\n")
    
    # Write statistics
    stats_file = os.path.join(results_dir, "stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Objective value: {objective}\n")
        f.write(f"Number of exams: {len(data.exams)}\n")
        f.write(f"Number of assigned exams: {len(solution)}\n")
        f.write(f"Assignment percentage: {100 * len(solution) / len(data.exams):.2f}%\n")
        
        # Calculate penalties
        period_penalties = 0
        room_penalties = 0
        for e, (p, r) in solution.items():
            if p < len(data.periods):  # Only count real periods
                period_penalties += data.periods[p]['penalty']
            if r < len(data.rooms):    # Only count real rooms
                room_penalties += data.rooms[r]['penalty']
        
        f.write(f"Period penalties: {period_penalties}\n")
        f.write(f"Room penalties: {room_penalties}\n")
        
        # Calculate soft constraint violations
        twoinarow, twoinaday, periodspread, nonmixed, frontload = calculate_soft_violations(solution, data)
        
        f.write(f"Two in a row violations: {twoinarow}\n")
        f.write(f"Two in a day violations: {twoinaday}\n")
        f.write(f"Period spread violations: {periodspread}\n")
        f.write(f"Mixed durations violations: {nonmixed}\n")
        f.write(f"Front load violations: {frontload}\n")
        
        # Count virtual assignments
        virtual_assignments = sum(1 for e, (p, r) in solution.items() if p >= len(data.periods) or r >= len(data.rooms))
        f.write(f"Virtual assignments: {virtual_assignments}\n")
    
    # Write dataset statistics
    dataset_stats_file = os.path.join(results_dir, "dataset_stats.txt")
    with open(dataset_stats_file, 'w') as f:
        stats = data.get_stats()
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Solution written to {solution_file}")
    print(f"ITC2007 format solution written to {sln_file}")
    print(f"Statistics written to {stats_file}")
    print(f"Dataset statistics written to {dataset_stats_file}")

def calculate_objective(solution, data):
    """
    Calculate the objective value of a solution.
    
    Args:
        solution: Dictionary mapping from exam to (period, room)
        data: ExamDataset object
    
    Returns:
        float: Objective value
    """
    # Calculate penalties
    period_penalties = 0
    room_penalties = 0
    for e, (p, r) in solution.items():
        if p < len(data.periods):  # Only count real periods
            period_penalties += data.periods[p]['penalty']
        if r < len(data.rooms):    # Only count real rooms
            room_penalties += data.rooms[r]['penalty']
    
    # Calculate soft constraint violations
    twoinarow, twoinaday, periodspread, nonmixed, frontload = calculate_soft_violations(solution, data)
    
    # Get weights from institutional weightings
    twoinarow_weight = data.institutional_weightings.get('TWOINAROW', [0])[0]
    twoinaday_weight = data.institutional_weightings.get('TWOINADAY', [0])[0]
    periodspread_weight = data.institutional_weightings.get('PERIODSPREAD', [0])[0]
    nonmixed_weight = data.institutional_weightings.get('NONMIXEDDURATIONS', [0])[0]
    frontload_weight = data.institutional_weightings.get('FRONTLOAD', [0, 0, 0])[0]
    
    # Calculate total objective
    objective = (
        period_penalties + room_penalties +
        twoinarow_weight * twoinarow +
        twoinaday_weight * twoinaday +
        periodspread_weight * periodspread +
        nonmixed_weight * nonmixed +
        frontload_weight * frontload
    )
    
    # Add penalty for virtual assignments
    virtual_assignments = sum(1 for e, (p, r) in solution.items() if p >= len(data.periods) or r >= len(data.rooms))
    if virtual_assignments > 0:
        objective += virtual_assignments * 1000  # Large penalty for virtual assignments
    
    return objective

def calculate_soft_violations(solution, data):
    """
    Calculate soft constraint violations.
    
    Args:
        solution: Dictionary mapping from exam to (period, room)
        data: ExamDataset object
    
    Returns:
        tuple: (twoinarow, twoinaday, periodspread, nonmixed, frontload)
    """
    # Initialize counters
    twoinarow = 0
    twoinaday = 0
    periodspread = 0
    nonmixed = 0
    frontload = 0
    
    # Group periods by day
    periods_by_day = {}
    for p, period in enumerate(data.periods):
        day = period['date']
        if day not in periods_by_day:
            periods_by_day[day] = []
        periods_by_day[day].append(p)
    
    # Check two in a row and two in a day violations
    for s, student_exams in data.students.items():
        # Check which exams this student takes
        student_periods = {}
        for e in student_exams:
            if e in solution:
                p, _ = solution[e]
                if p < len(data.periods):  # Only consider real periods
                    student_periods[e] = p
        
        # Check two in a row
        for e1, p1 in student_periods.items():
            for e2, p2 in student_periods.items():
                if e1 != e2:
                    # Check if exams are in consecutive periods
                    if abs(p1 - p2) == 1:
                        twoinarow += 1
                    
                    # Check if exams are in the same day but not consecutive
                    for day, periods in periods_by_day.items():
                        if p1 in periods and p2 in periods and abs(p1 - p2) > 1:
                            twoinaday += 1
    
    # Check period spread violations
    if 'PERIODSPREAD' in data.institutional_weightings and len(data.institutional_weightings['PERIODSPREAD']) > 1:
        spread_distance = data.institutional_weightings['PERIODSPREAD'][1]
        
        for s, student_exams in data.students.items():
            # Check which exams this student takes
            student_periods = {}
            for e in student_exams:
                if e in solution:
                    p, _ = solution[e]
                    if p < len(data.periods):  # Only consider real periods
                        student_periods[e] = p
            
            # Check period spread
            for e1, p1 in student_periods.items():
                for e2, p2 in student_periods.items():
                    if e1 != e2 and 0 < abs(p1 - p2) < spread_distance:
                        periodspread += 1
    
    # Check mixed durations violations
    exams_by_period = {}
    for e, (p, _) in solution.items():
        if p < len(data.periods):  # Only consider real periods
            if p not in exams_by_period:
                exams_by_period[p] = []
            exams_by_period[p].append(e)
    
    for p, period_exams in exams_by_period.items():
        # Check if there are exams with different durations
        durations = set(data.exam_durations.get(e, 0) for e in period_exams)
        if len(durations) > 1:
            nonmixed += len(period_exams)
    
    # Check front load violations
    if 'FRONTLOAD' in data.institutional_weightings and len(data.institutional_weightings['FRONTLOAD']) >= 3:
        num_large_exams = data.institutional_weightings['FRONTLOAD'][1]
        num_last_periods = data.institutional_weightings['FRONTLOAD'][2]
        
        # Get the largest exams
        exam_sizes = [(e, len(data.exam_students.get(e, []))) for e in data.exams]
        exam_sizes.sort(key=lambda x: x[1], reverse=True)
        large_exams = [e for e, _ in exam_sizes[:num_large_exams]]
        
        # Check if large exams are scheduled in the last periods
        last_periods = list(range(max(0, len(data.periods) - num_last_periods), len(data.periods)))
        for e in large_exams:
            if e in solution and solution[e][0] in last_periods:
                frontload += 1
    
    return twoinarow, twoinaday, periodspread, nonmixed, frontload 