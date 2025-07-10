import re
import math
import numpy as np
from collections import defaultdict

class ExamDataset:
    def __init__(self, filename):
        self.filename = filename
        self.exams = []  # List of exam IDs
        self.exam_durations = {}  # exam_id -> duration
        self.students = defaultdict(list)  # student_id -> list of exams
        self.exam_students = defaultdict(list)  # exam_id -> list of students
        self.periods = []  # List of dicts: {date, time, duration, penalty}
        self.rooms = []  # List of dicts: {capacity, penalty}
        self.period_constraints = []  # List of tuples (exam1, type, exam2)
        self.room_constraints = []  # Not used in this dataset
        self.institutional_weightings = {}  # Weights for soft constraints
        
        # Additional attributes based on C++ implementation
        self.conflict_matrix = None  # Will be computed after parsing
        self.conflict_list = None  # List of exams each exam conflicts with
        self.valid_periods = None  # Mask of valid periods for each exam
        self.valid_rooms = None  # Mask of valid rooms for each exam
        self.components = None  # Connected components in the conflict graph
        self.unique_exam_durations = None  # List of unique exam durations
        self.front_load_exams = None  # List of exams for front loading
        
        # Parse the dataset
        self._parse()
        
        # Compute additional data
        self._compute_conflict_matrix()
        self._compute_valid_assignments()
        self._compute_components()
        self._compute_unique_durations()
        self._compute_front_load_exams()
        
    def _parse(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Parse the file section by section
        section = None
        student_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.startswith('[Exams:'):
                section = 'exams'
                match = re.search(r'\[Exams:(\d+)\]', line)
                if match:
                    num_exams = int(match.group(1))
                continue
            elif line.startswith('[Periods:'):
                section = 'periods'
                continue
            elif line.startswith('[Rooms:'):
                section = 'rooms'
                continue
            elif line.startswith('[PeriodHardConstraints]'):
                section = 'period_constraints'
                continue
            elif line.startswith('[RoomHardConstraints]'):
                section = 'room_constraints'
                continue
            elif line.startswith('[InstitutionalWeightings]'):
                section = 'institutional_weightings'
                continue
            
            # Parse each section
            if section == 'exams':
                # Each line represents an exam with duration and student enrollments
                parts = line.split(',')
                if len(parts) >= 1:
                    duration = int(parts[0].strip())
                    exam_id = len(self.exams)
                    self.exams.append(exam_id)
                    self.exam_durations[exam_id] = duration
                    
                    # Process student enrollments for this exam
                    for i in range(1, len(parts)):
                        student = int(parts[i].strip())
                        self.students[student].append(exam_id)
                        self.exam_students[exam_id].append(student)
                    
                    student_id += 1
                    
            elif section == 'periods':
                # Format: date, time, duration, penalty
                parts = line.split(',')
                if len(parts) >= 4:
                    period = {
                        'date': parts[0].strip(),
                        'time': parts[1].strip(),
                        'duration': int(parts[2].strip()),
                        'penalty': int(parts[3].strip())
                    }
                    self.periods.append(period)
                    
            elif section == 'rooms':
                # Format: capacity, penalty
                parts = line.split(',')
                if len(parts) >= 2:
                    room = {
                        'capacity': int(parts[0].strip()),
                        'penalty': int(parts[1].strip())
                    }
                    self.rooms.append(room)
                    
            elif section == 'period_constraints':
                # Format: exam1, constraint_type, exam2
                parts = line.split(',')
                if len(parts) >= 3:
                    exam1 = int(parts[0].strip())
                    constraint_type = parts[1].strip()
                    exam2 = int(parts[2].strip())
                    self.period_constraints.append((exam1, constraint_type, exam2))
                    
            elif section == 'room_constraints':
                # Not used in this dataset
                pass
                
            elif section == 'institutional_weightings':
                # Format: constraint_type, weight(s)
                parts = line.split(',')
                if len(parts) >= 2:
                    constraint_type = parts[0].strip()
                    weights = [int(w.strip()) for w in parts[1:]]
                    self.institutional_weightings[constraint_type] = weights
    
    def _compute_conflict_matrix(self):
        """Compute the conflict matrix between exams (similar to C++ implementation)"""
        num_exams = len(self.exams)
        self.conflict_matrix = np.zeros((num_exams, num_exams), dtype=int)
        self.conflict_list = [[] for _ in range(num_exams)]
        
        # For each student, add conflicts between their exams
        for student_id, student_exams in self.students.items():
            for i in range(len(student_exams)):
                for j in range(i+1, len(student_exams)):
                    e1, e2 = student_exams[i], student_exams[j]
                    self.conflict_matrix[e1, e2] += 1
                    self.conflict_matrix[e2, e1] += 1
        
        # Create conflict lists for each exam
        for e1 in range(num_exams):
            for e2 in range(num_exams):
                if self.conflict_matrix[e1, e2] > 0 and e1 != e2:
                    self.conflict_list[e1].append(e2)
        
        # Compute conflict density (CD) as in C++ implementation
        total_conflicts = sum(len(conflicts) for conflicts in self.conflict_list) // 2
        max_possible_conflicts = (num_exams * (num_exams - 1)) // 2
        self.conflict_density = total_conflicts / max_possible_conflicts if max_possible_conflicts > 0 else 0
    
    def _compute_valid_assignments(self):
        """Compute valid periods and rooms for each exam"""
        num_exams = len(self.exams)
        num_periods = len(self.periods)
        num_rooms = len(self.rooms)
        
        self.valid_periods = np.ones((num_exams, num_periods), dtype=bool)
        self.valid_rooms = np.ones((num_exams, num_rooms), dtype=bool)
        
        # Check period durations
        for e in range(num_exams):
            for p in range(num_periods):
                if self.exam_durations[e] > self.periods[p]['duration']:
                    self.valid_periods[e, p] = False
        
        # Check room capacities
        for e in range(num_exams):
            num_students = len(self.exam_students[e])
            for r in range(num_rooms):
                if num_students > self.rooms[r]['capacity']:
                    self.valid_rooms[e, r] = False
        
        # Process period constraints
        for exam1, constraint_type, exam2 in self.period_constraints:
            if constraint_type == "AFTER":
                # exam2 must be scheduled after exam1
                pass  # This is handled in the solver
            elif constraint_type == "EXAM_COINCIDENCE":
                # exam1 and exam2 must be scheduled in the same period
                # We'll make their valid periods the same (intersection)
                common_valid = self.valid_periods[exam1] & self.valid_periods[exam2]
                self.valid_periods[exam1] = common_valid
                self.valid_periods[exam2] = common_valid
            elif constraint_type == "EXCLUSION":
                # Already handled by conflict matrix
                pass
        
        # Create lists of valid periods and rooms for each exam
        self.valid_period_list = [np.where(self.valid_periods[e])[0].tolist() for e in range(num_exams)]
        self.valid_room_list = [np.where(self.valid_rooms[e])[0].tolist() for e in range(num_exams)]
    
    def _compute_components(self):
        """Find connected components in the conflict graph (like in C++ implementation)"""
        num_exams = len(self.exams)
        visited = [False] * num_exams
        self.components = []
        
        for e in range(num_exams):
            if not visited[e]:
                component = []
                self._dfs(e, visited, component)
                if component:
                    # Sort by exam size (number of students) as in C++ implementation
                    component.sort(key=lambda x: len(self.exam_students[x]), reverse=True)
                    self.components.append(component)
        
        # Calculate ExR (Exams per Room) as in C++ implementation
        self.exams_per_room = num_exams / len(self.rooms) if self.rooms else 0
        
        # Calculate SxE (Students per Exam) as in C++ implementation
        total_enrollments = sum(len(students) for students in self.exam_students.values())
        self.students_per_exam = total_enrollments / num_exams if num_exams > 0 else 0
        
        # Calculate SCap (Student Capacity)
        total_capacity = sum(room['capacity'] for room in self.rooms) * len(self.periods)
        self.student_capacity = total_enrollments / total_capacity if total_capacity > 0 else 0
    
    def _dfs(self, exam, visited, component):
        """Depth-first search for finding connected components"""
        visited[exam] = True
        component.append(exam)
        
        for neighbor in self.conflict_list[exam]:
            if not visited[neighbor]:
                self._dfs(neighbor, visited, component)
    
    def _compute_unique_durations(self):
        """Compute unique exam durations (as in C++ implementation)"""
        self.unique_exam_durations = sorted(set(self.exam_durations.values()))
        
        # Create duration index for each exam
        self.exam_duration_index = {}
        for e in self.exams:
            duration = self.exam_durations[e]
            self.exam_duration_index[e] = self.unique_exam_durations.index(duration)
    
    def _compute_front_load_exams(self):
        """Identify exams for front loading (largest exams)"""
        if 'FRONTLOAD' in self.institutional_weightings:
            frontload_params = self.institutional_weightings['FRONTLOAD']
            if len(frontload_params) >= 2:
                num_large_exams = frontload_params[1]
                
                # Sort exams by number of students
                exam_sizes = [(e, len(self.exam_students[e])) for e in self.exams]
                exam_sizes.sort(key=lambda x: x[1], reverse=True)
                
                # Get the largest exams
                self.front_load_exams = [e for e, _ in exam_sizes[:num_large_exams]]
            else:
                self.front_load_exams = []
        else:
            self.front_load_exams = []
    
    def get_stats(self):
        """
        Get statistics about the dataset.
        
        Returns:
            dict: Statistics about the dataset
        """
        stats = {}
        stats['num_exams'] = len(self.exams)
        stats['num_students'] = len(self.students)
        stats['num_periods'] = len(self.periods)
        stats['num_rooms'] = len(self.rooms)
        stats['num_period_constraints'] = len(self.period_constraints)
        stats['num_room_constraints'] = len(self.room_constraints)
        stats['num_components'] = len(self.components) if self.components else 0
        stats['conflict_density'] = self.conflict_density if hasattr(self, 'conflict_density') else 0
        stats['exams_per_room'] = self.exams_per_room if hasattr(self, 'exams_per_room') else 0
        stats['students_per_exam'] = self.students_per_exam if hasattr(self, 'students_per_exam') else 0
        stats['student_capacity'] = self.student_capacity if hasattr(self, 'student_capacity') else 0
        
        # Format institutional weightings for better readability
        weightings_str = "{\n"
        for constraint, weights in self.institutional_weightings.items():
            weightings_str += f"  '{constraint}': {weights},\n"
        weightings_str += "}"
        stats['institutional_weightings'] = weightings_str
        
        return stats
    
    def get_conflict_count(self, exam1, exam2):
        """Get the number of conflicts between two exams"""
        if self.conflict_matrix is not None:
            return self.conflict_matrix[exam1, exam2]
        return 0
    
    def get_conflicts_for_exam(self, exam):
        """Get all exams that conflict with the given exam"""
        if self.conflict_list is not None:
            return self.conflict_list[exam]
        return []
    
    def is_valid_period(self, exam, period):
        """Check if a period is valid for an exam"""
        if self.valid_periods is not None:
            return self.valid_periods[exam, period]
        return True
    
    def is_valid_room(self, exam, room):
        """Check if a room is valid for an exam"""
        if self.valid_rooms is not None:
            return self.valid_rooms[exam, room]
        return True
    
    def get_valid_periods(self, exam):
        """Get all valid periods for an exam"""
        if hasattr(self, 'valid_period_list'):
            return self.valid_period_list[exam]
        return list(range(len(self.periods)))
    
    def get_valid_rooms(self, exam):
        """Get all valid rooms for an exam"""
        if hasattr(self, 'valid_room_list'):
            return self.valid_room_list[exam]
        return list(range(len(self.rooms)))
    
    def get_exam_count(self):
        return len(self.exams)

    def get_student_count(self):
        return len(self.students)

    def get_period_count(self):
        return len(self.periods)

    def get_room_count(self):
        return len(self.rooms)
    
    def get_component_count(self):
        return len(self.components) if self.components else 0
    
    def get_component(self, index):
        """Get exams in a specific component"""
        if self.components and 0 <= index < len(self.components):
            return self.components[index]
        return []
    
    def __str__(self):
        """Return a string representation of the dataset."""
        stats = self.get_stats()
        return (
            f"ITC 2007 Exam Dataset: {self.filename}\n"
            f"Number of exams: {stats['num_exams']}\n"
            f"Number of students: {stats['num_students']}\n"
            f"Number of periods: {stats['num_periods']}\n"
            f"Number of rooms: {stats['num_rooms']}\n"
            f"Number of components: {stats['num_components']}\n"
            f"Conflict density: {stats['conflict_density']:.4f}\n"
            f"Exams per room: {stats['exams_per_room']:.2f}\n"
            f"Students per exam: {stats['students_per_exam']:.2f}\n"
            f"Student capacity ratio: {stats['student_capacity']:.4f}\n"
            f"Institutional weightings: {stats['institutional_weightings']}"
        ) 