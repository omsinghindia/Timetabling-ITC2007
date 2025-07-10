import os
os.environ["GRB_LICENSE_FILE"] = "C:/Users/hp/gurobi.lic"

import gurobipy as gp
from gurobipy import GRB
from utils.parser import ExamDataset
import math
import time
import random
import itertools

def create_mock_solution(dataset_path, results_dir):
    """
    Create a mock solution for testing purposes when Gurobi is not available.
    This function creates a feasible solution without optimization.
    
    Args:
        dataset_path: Path to the dataset file
        results_dir: Directory to write results
    """
    print("Creating mock solution for testing purposes...")
    
    # Load dataset
    start_time = time.time()
    print(f"Loading dataset: {dataset_path}")
    data = ExamDataset(dataset_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Extract data
    exams = data.exams
    students = data.students
    exam_students = data.exam_students
    periods = data.periods
    rooms = data.rooms
    
    num_exams = len(exams)
    num_periods = len(periods)
    num_rooms = len(rooms)
    
    # Create a simple greedy solution
    # For each exam, find the first available period and room
    solution = {}
    period_room_usage = {}  # (period, room) -> exam
    student_period_usage = {}  # (student, period) -> True/False
    
    for e in sorted(exams):
        assigned = False
        
        # Try each period
        for p in range(num_periods):
            # Check if any student has an exam in this period
            student_conflict = False
            for s in exam_students[e]:
                if (s, p) in student_period_usage:
                    student_conflict = True
                    break
            
            if student_conflict:
                continue
                
            # Find a room with sufficient capacity
            for r in range(num_rooms):
                if (p, r) not in period_room_usage and rooms[r]['capacity'] >= len(exam_students[e]):
                    # Assign the exam to this period and room
                    solution[e] = (p, r)
                    period_room_usage[(p, r)] = e
                    
                    # Mark all students as having an exam in this period
                    for s in exam_students[e]:
                        student_period_usage[(s, p)] = True
                        
                    assigned = True
                    break
                    
            if assigned:
                break
                
        if not assigned:
            print(f"Warning: Could not assign exam {e} to any period and room.")
            # Assign to a random period and room (not feasible, but for testing only)
            p = random.randint(0, num_periods - 1)
            r = random.randint(0, num_rooms - 1)
            solution[e] = (p, r)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Write solution to file
    solution_file = os.path.join(results_dir, "solution.txt")
    with open(solution_file, 'w') as f:
        f.write(f"Mock solution (not optimized)\n\n")
        f.write("Exam assignments:\n")
        for e in sorted(exams):
            p, r = solution[e]
            f.write(f"Exam {e}: Period {p}, Room {r}\n")
    
    # Write ITC2007 format solution file (.sln)
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    sln_file = os.path.join(results_dir, f"{dataset_name}.sln")
    
    with open(sln_file, 'w') as f:
        for e in sorted(exams):
            p, r = solution[e]
            f.write(f"{p}, {r}\r\n\r\n")
    
    # Write statistics
    stats_file = os.path.join(results_dir, "stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Status: Mock solution (not optimized)\n")
        f.write(f"Objective value: N/A\n")
        f.write(f"Runtime: {time.time() - start_time:.2f} seconds\n")
    
    # Write dataset statistics
    dataset_stats_file = os.path.join(results_dir, "dataset_stats.txt")
    with open(dataset_stats_file, 'w') as f:
        stats = data.get_stats()
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Mock solution written to {solution_file}")
    print(f"ITC2007 format solution written to {sln_file}")
    print(f"Statistics written to {stats_file}")
    print(f"Dataset statistics written to {dataset_stats_file}")
    
    return "Mock solution created"

def solve_itc2007(dataset_path, results_dir, time_limit=None, debug=False):
    """
    Solve the ITC 2007 Examination Timetabling Problem using Gurobi.
    
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
    students = data.students
    exam_students = data.exam_students
    periods = data.periods
    rooms = data.rooms
    exam_durations = data.exam_durations
    period_constraints = data.period_constraints
    institutional_weightings = data.institutional_weightings
    
    num_exams = len(exams)
    num_periods = len(periods)
    num_rooms = len(rooms)
    
    if debug:
        print(f"Problem size: {num_exams} exams, {num_periods} periods, {num_rooms} rooms")
        print(f"Number of students: {len(students)}")
        print(f"Number of period constraints: {len(period_constraints)}")
        print(f"Institutional weightings: {institutional_weightings}")
    
    # Pre-compute conflicting exams (exams with common students)
    if debug:
        print("Pre-computing conflicting exams...")
    
    conflict_matrix = {}  # (e1, e2) -> number of common students
    for s, student_exams in students.items():
        for e1, e2 in itertools.combinations(student_exams, 2):
            if (e1, e2) not in conflict_matrix:
                conflict_matrix[(e1, e2)] = 1
                conflict_matrix[(e2, e1)] = 1
            else:
                conflict_matrix[(e1, e2)] += 1
                conflict_matrix[(e2, e1)] += 1
    
    # Create model
    model = gp.Model("ITC2007_Exam_Timetabling")
    
    # Set time limit if specified
    if time_limit:
        model.setParam('TimeLimit', time_limit)
    
    # Set additional parameters for debugging
    if debug:
        model.setParam('OutputFlag', 1)  # Enable Gurobi output
        model.setParam('LogToConsole', 1)
    else:
        model.setParam('OutputFlag', 0)  # Disable Gurobi output
        model.setParam('LogToConsole', 0)
    
    # Set MIP focus to finding feasible solutions quickly
    model.setParam('MIPFocus', 1)
    
    if debug:
        print("Creating decision variables...")
    
    # Decision variables
    # x[e,p,r] = 1 if exam e is scheduled in period p and room r
    x = model.addVars([(e, p, r) for e in exams for p in range(num_periods) for r in range(num_rooms)], 
                     vtype=GRB.BINARY, name="x")
    
    # y[e,p] = 1 if exam e is scheduled in period p
    y = model.addVars([(e, p) for e in exams for p in range(num_periods)], 
                     vtype=GRB.BINARY, name="y")
    
    if debug:
        print(f"Created {len(x)} x variables and {len(y)} y variables")
        print("Linking x and y variables...")
    
    # Link x and y variables
    for e in exams:
        for p in range(num_periods):
            model.addConstr(y[e, p] == gp.quicksum(x[e, p, r] for r in range(num_rooms)), 
                           name=f"link_{e}_{p}")
    
    if debug:
        print("Adding hard constraints...")
    
    # Hard Constraints
    
    # HC1: Each exam must be scheduled exactly once
    for e in exams:
        model.addConstr(gp.quicksum(y[e, p] for p in range(num_periods)) == 1, 
                       name=f"exam_assignment_{e}")
    
    if debug:
        print("Added exam assignment constraints")
    
    # HC2: Student conflicts - no student can take two exams at the same time
    # We can use the pre-computed conflict matrix to add these constraints more efficiently
    conflict_count = 0
    for (e1, e2), count in conflict_matrix.items():
        if e1 < e2:  # Only add constraint once for each pair
            for p in range(num_periods):
                model.addConstr(y[e1, p] + y[e2, p] <= 1, 
                               name=f"conflict_{e1}_{e2}_{p}")
                conflict_count += 1
    
    if debug:
        print(f"Added {conflict_count} conflict constraints")
    
    # HC3: Room capacity - the room must be large enough for the exam
    room_capacity_count = 0
    for e in exams:
        for p in range(num_periods):
            for r in range(num_rooms):
                num_students = len(exam_students[e])
                room_capacity = rooms[r]['capacity']
                if num_students > room_capacity:
                    model.addConstr(x[e, p, r] == 0, 
                                  name=f"room_capacity_{e}_{p}_{r}")
                    room_capacity_count += 1
    
    if debug:
        print(f"Added {room_capacity_count} room capacity constraints")
    
    # HC4: Period duration - the exam duration must not exceed the period duration
    period_duration_count = 0
    for e in exams:
        for p in range(num_periods):
            if exam_durations[e] > periods[p]['duration']:
                model.addConstr(y[e, p] == 0, 
                               name=f"period_duration_{e}_{p}")
                period_duration_count += 1
    
    if debug:
        print(f"Added {period_duration_count} period duration constraints")
    
    # HC5: Period hard constraints
    period_constraint_count = 0
    for exam1, constraint_type, exam2 in period_constraints:
        if constraint_type == "AFTER":
            # exam2 must be scheduled after exam1
            for p1 in range(num_periods):
                for p2 in range(p1 + 1):  # p2 <= p1
                    model.addConstr(y[exam1, p1] + y[exam2, p2] <= 1, 
                                   name=f"after_{exam1}_{exam2}_{p1}_{p2}")
                    period_constraint_count += 1
        elif constraint_type == "EXAM_COINCIDENCE":
            # exam1 and exam2 must be scheduled in the same period
            for p in range(num_periods):
                model.addConstr(y[exam1, p] == y[exam2, p], 
                               name=f"coincidence_{exam1}_{exam2}_{p}")
                period_constraint_count += 1
        elif constraint_type == "EXCLUSION":
            # exam1 and exam2 must be scheduled in different periods
            for p in range(num_periods):
                model.addConstr(y[exam1, p] + y[exam2, p] <= 1, 
                               name=f"exclusion_{exam1}_{exam2}_{p}")
                period_constraint_count += 1
    
    if debug:
        print(f"Added {period_constraint_count} period hard constraints")
        print("Adding soft constraints...")
    
    # Soft Constraints (Institutional Weightings)
    
    # SC1: Two Exams in a Row (TWOINAROW)
    twoinarow_weight = institutional_weightings.get('TWOINAROW', [0])[0]
    twoinarow_penalty_vars = []
    if twoinarow_weight > 0:
        if debug:
            print(f"Adding TWOINAROW constraints with weight {twoinarow_weight}")
        
        # Only consider pairs of exams that have students in common
        for (e1, e2), count in conflict_matrix.items():
            if e1 < e2:  # Only add penalty once for each pair
                for p in range(num_periods - 1):
                    if p % 2 == 0:  # Only consider morning periods
                        twoinarow = model.addVar(vtype=GRB.BINARY, name=f"twoinarow_{e1}_{e2}_{p}")
                        model.addConstr(twoinarow >= y[e1, p] + y[e2, p+1] - 1, 
                                      name=f"twoinarow_constr_{e1}_{e2}_{p}")
                        # Weight by the number of affected students
                        twoinarow_penalty_vars.append(count * twoinarow)
        
        if debug and twoinarow_penalty_vars:
            print(f"Added {len(twoinarow_penalty_vars)} TWOINAROW penalty variables")
    
    # SC2: Two Exams in a Day (TWOINADAY)
    twoinaday_weight = institutional_weightings.get('TWOINADAY', [0])[0]
    twoinaday_penalty_vars = []
    if twoinaday_weight > 0:
        if debug:
            print(f"Adding TWOINADAY constraints with weight {twoinaday_weight}")
        
        # Only consider pairs of exams that have students in common
        for (e1, e2), count in conflict_matrix.items():
            if e1 < e2:  # Only add penalty once for each pair
                for day in range(num_periods // 2):  # Assuming 2 periods per day
                    morning = day * 2
                    afternoon = day * 2 + 1
                    if morning < num_periods and afternoon < num_periods:
                        twoinaday = model.addVar(vtype=GRB.BINARY, name=f"twoinaday_{e1}_{e2}_{day}")
                        model.addConstr(twoinaday >= y[e1, morning] + y[e2, afternoon] - 1, 
                                      name=f"twoinaday_constr_{e1}_{e2}_{day}")
                        # Weight by the number of affected students
                        twoinaday_penalty_vars.append(count * twoinaday)
        
        if debug and twoinaday_penalty_vars:
            print(f"Added {len(twoinaday_penalty_vars)} TWOINADAY penalty variables")
    
    # SC3: Period Spread (PERIODSPREAD)
    periodspread_weight = institutional_weightings.get('PERIODSPREAD', [0])[0]
    periodspread_penalty_vars = []
    if periodspread_weight > 0:
        if debug:
            print(f"Adding PERIODSPREAD constraints with weight {periodspread_weight}")
        
        # Only consider pairs of exams that have students in common
        for (e1, e2), count in conflict_matrix.items():
            if e1 < e2:  # Only add penalty once for each pair
                for p1 in range(num_periods):
                    for p2 in range(p1 + 1, min(p1 + 6, num_periods)):  # Within 5 periods
                        periodspread = model.addVar(vtype=GRB.BINARY, name=f"periodspread_{e1}_{e2}_{p1}_{p2}")
                        model.addConstr(periodspread >= y[e1, p1] + y[e2, p2] - 1, 
                                      name=f"periodspread_constr_{e1}_{e2}_{p1}_{p2}")
                        # Weight by the number of affected students
                        periodspread_penalty_vars.append(count * periodspread)
        
        if debug and periodspread_penalty_vars:
            print(f"Added {len(periodspread_penalty_vars)} PERIODSPREAD penalty variables")
    
    # SC4: Mixed Durations (NONMIXEDDURATIONS)
    nonmixed_weight = institutional_weightings.get('NONMIXEDDURATIONS', [0])[0]
    nonmixed_penalty_vars = []
    if nonmixed_weight > 0:
        if debug:
            print(f"Adding NONMIXEDDURATIONS constraints with weight {nonmixed_weight}")
        
        # Group exams by duration
        exams_by_duration = {}
        for e in exams:
            duration = exam_durations[e]
            if duration not in exams_by_duration:
                exams_by_duration[duration] = []
            exams_by_duration[duration].append(e)
        
        # For each period, add penalties for exams with different durations
        for p in range(num_periods):
            for duration1, exams1 in exams_by_duration.items():
                for duration2, exams2 in exams_by_duration.items():
                    if duration1 < duration2:  # Only consider each pair once
                        nonmixed = model.addVar(vtype=GRB.BINARY, name=f"nonmixed_{duration1}_{duration2}_{p}")
                        
                        # If at least one exam from each duration group is scheduled in this period,
                        # the penalty variable should be 1
                        model.addConstr(
                            nonmixed >= 
                            gp.quicksum(y[e1, p] for e1 in exams1) / len(exams1) +
                            gp.quicksum(y[e2, p] for e2 in exams2) / len(exams2) - 1,
                            name=f"nonmixed_constr_{duration1}_{duration2}_{p}"
                        )
                        
                        nonmixed_penalty_vars.append(nonmixed)
        
        if debug and nonmixed_penalty_vars:
            print(f"Added {len(nonmixed_penalty_vars)} NONMIXEDDURATIONS penalty variables")
    
    # SC5: Front Load (FRONTLOAD)
    frontload_params = institutional_weightings.get('FRONTLOAD', [0, 0, 0])
    frontload_penalty_vars = []
    if len(frontload_params) >= 3 and frontload_params[0] > 0:
        frontload_weight = frontload_params[0]
        num_large_exams = frontload_params[1]
        num_last_periods = frontload_params[2]
        
        if debug:
            print(f"Adding FRONTLOAD constraints with weight {frontload_weight}, {num_large_exams} large exams, {num_last_periods} last periods")
        
        # Identify large exams (those with the most students)
        exam_sizes = [(e, len(exam_students[e])) for e in exams]
        exam_sizes.sort(key=lambda x: x[1], reverse=True)
        large_exams = [e for e, _ in exam_sizes[:num_large_exams]]
        
        for e in large_exams:
            for p in range(num_periods - num_last_periods, num_periods):
                frontload_penalty_vars.append(y[e, p])
        
        if debug and frontload_penalty_vars:
            print(f"Added {len(frontload_penalty_vars)} FRONTLOAD penalty variables")
    
    if debug:
        print("Setting up objective function...")
    
    # Objective function
    obj = gp.LinExpr()
    
    # Add penalty for each soft constraint
    if twoinarow_weight > 0 and twoinarow_penalty_vars:
        obj.add(twoinarow_weight * gp.quicksum(twoinarow_penalty_vars))
    
    if twoinaday_weight > 0 and twoinaday_penalty_vars:
        obj.add(twoinaday_weight * gp.quicksum(twoinaday_penalty_vars))
    
    if periodspread_weight > 0 and periodspread_penalty_vars:
        obj.add(periodspread_weight * gp.quicksum(periodspread_penalty_vars))
    
    if nonmixed_weight > 0 and nonmixed_penalty_vars:
        obj.add(nonmixed_weight * gp.quicksum(nonmixed_penalty_vars))
    
    if len(frontload_params) >= 3 and frontload_params[0] > 0 and frontload_penalty_vars:
        obj.add(frontload_params[0] * gp.quicksum(frontload_penalty_vars))
    
    # Add period penalties
    for e in exams:
        for p in range(num_periods):
            period_penalty = periods[p]['penalty']
            if period_penalty > 0:
                obj.addTerms(period_penalty, y[e, p])
    
    # Add room penalties
    for e in exams:
        for p in range(num_periods):
            for r in range(num_rooms):
                room_penalty = rooms[r]['penalty']
                if room_penalty > 0:
                    obj.addTerms(room_penalty, x[e, p, r])
    
    # Set objective
    model.setObjective(obj, GRB.MINIMIZE)
    
    if debug:
        print(f"Model has {model.NumVars} variables and {model.NumConstrs} constraints")
        print("Starting optimization...")
    
    # Optimize
    model.optimize()
    
    if debug:
        print(f"Optimization status: {model.Status}")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.ObjVal}")
        elif model.Status == GRB.TIME_LIMIT:
            print(f"Time limit reached with objective value: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
        elif model.Status == GRB.INFEASIBLE:
            print("Model is infeasible")
        elif model.Status == GRB.UNBOUNDED:
            print("Model is unbounded")
    
    # Check if a solution was found
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        if model.SolCount > 0:
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Write solution to file
            solution_file = os.path.join(results_dir, "solution.txt")
            with open(solution_file, 'w') as f:
                f.write(f"Objective value: {model.ObjVal}\n\n")
                f.write("Exam assignments:\n")
                for e in exams:
                    for p in range(num_periods):
                        for r in range(num_rooms):
                            if x[e, p, r].X > 0.5:
                                f.write(f"Exam {e}: Period {p}, Room {r}\n")
            
            # Write ITC2007 format solution file (.sln)
            # The solution file must have one line per exam in the same order as the input file
            # Each line contains: period_number, room_number
            dataset_name = os.path.basename(dataset_path).split('.')[0]
            sln_file = os.path.join(results_dir, f"{dataset_name}.sln")
            write_itc2007_solution(sln_file, model, x, exams, num_periods, num_rooms)
            
            # Write statistics
            stats_file = os.path.join(results_dir, "stats.txt")
            with open(stats_file, 'w') as f:
                f.write(f"Status: {model.Status}\n")
                f.write(f"Objective value: {model.ObjVal}\n")
                f.write(f"Runtime: {model.Runtime} seconds\n")
                f.write(f"MIP gap: {model.MIPGap}\n")
                f.write(f"Number of variables: {model.NumVars}\n")
                f.write(f"Number of constraints: {model.NumConstrs}\n")
                
                # Write penalty breakdown
                f.write("\nPenalty breakdown:\n")
                
                if twoinarow_weight > 0 and twoinarow_penalty_vars:
                    penalty_val = sum(var.X for var in twoinarow_penalty_vars)
                    f.write(f"TWOINAROW: {penalty_val} * {twoinarow_weight} = {penalty_val * twoinarow_weight}\n")
                
                if twoinaday_weight > 0 and twoinaday_penalty_vars:
                    penalty_val = sum(var.X for var in twoinaday_penalty_vars)
                    f.write(f"TWOINADAY: {penalty_val} * {twoinaday_weight} = {penalty_val * twoinaday_weight}\n")
                
                if periodspread_weight > 0 and periodspread_penalty_vars:
                    penalty_val = sum(var.X for var in periodspread_penalty_vars)
                    f.write(f"PERIODSPREAD: {penalty_val} * {periodspread_weight} = {penalty_val * periodspread_weight}\n")
                
                if nonmixed_weight > 0 and nonmixed_penalty_vars:
                    penalty_val = sum(var.X for var in nonmixed_penalty_vars)
                    f.write(f"NONMIXEDDURATIONS: {penalty_val} * {nonmixed_weight} = {penalty_val * nonmixed_weight}\n")
                
                if len(frontload_params) >= 3 and frontload_params[0] > 0 and frontload_penalty_vars:
                    penalty_val = sum(var.X for var in frontload_penalty_vars)
                    f.write(f"FRONTLOAD: {penalty_val} * {frontload_params[0]} = {penalty_val * frontload_params[0]}\n")
            
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
            
            return model.ObjVal
        else:
            print("No solution found within the time limit.")
            return None
    else:
        print(f"Optimization failed with status {model.status}")
        return None

def write_itc2007_solution(sln_file, model, x, exams, num_periods, num_rooms):
    """
    Write the solution in the ITC2007 format.
    
    Args:
        sln_file: Path to the output .sln file
        model: Gurobi model with solution
        x: Decision variables x[e,p,r]
        exams: List of exam IDs
        num_periods: Number of periods
        num_rooms: Number of rooms
    """
    with open(sln_file, 'w') as f:
        for e in sorted(exams):  # Ensure exams are in order
            period = -1
            room = -1
            for p in range(num_periods):
                for r in range(num_rooms):
                    if x[e, p, r].X > 0.5:
                        period = p
                        room = r
                        break
                if period != -1:
                    break
            
            # Write the period and room for this exam
            # Format: period, room with empty line between entries
            f.write(f"{period}, {room}\r\n\r\n")

def analyze_dataset(dataset_path):
    """
    Analyze the dataset without solving the problem.
    
    Args:
        dataset_path: Path to the dataset file
    """
    start_time = time.time()
    print(f"Loading dataset: {dataset_path}")
    data = ExamDataset(dataset_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Print statistics
    stats = data.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print institutional weightings
    print(f"Institutional weightings: {data.institutional_weightings}")
    
    return data