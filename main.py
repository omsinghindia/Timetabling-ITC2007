import os
import argparse
import time
from pathlib import Path

# Set Gurobi license path
os.environ["GRB_LICENSE_FILE"] = "C:/Users/hp/gurobi.lic"

def list_available_datasets():
    """List all available datasets in the datasets folder."""
    datasets = []
    for file in os.listdir('datasets'):
        if file.endswith('.exam'):
            datasets.append(file)
    return sorted(datasets)

def get_yes_no_input(prompt):
    """Get a yes/no input from the user."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Solve ITC 2007 Examination Timetabling Problem')
    parser.add_argument('--dataset', type=str, help='Dataset file name (must be in datasets/ directory)')
    parser.add_argument('--time-limit', type=int, help='Time limit in seconds')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze the dataset without solving')
    parser.add_argument('--mock', action='store_true', help='Create a mock solution without using Gurobi')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced solver')
    
    args = parser.parse_args()
    
    # If dataset is not provided, prompt the user to select one
    if not args.dataset:
        available_datasets = list_available_datasets()
        
        if not available_datasets:
            print("No datasets found in the datasets/ directory.")
            return
        
        print("Available datasets:")
        for i, dataset in enumerate(available_datasets):
            print(f"{i+1}. {dataset}")
        
        while True:
            try:
                choice = input("\nEnter the number of the dataset you want to use (or 'q' to quit): ")
                if choice.lower() in ['q', 'quit', 'exit']:
                    print("Exiting program.")
                    return
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_datasets):
                    args.dataset = available_datasets[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_datasets)}.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Prompt for enhanced solver if not specified
    if not args.enhanced and not args.analyze_only and not args.mock:
        args.enhanced = get_yes_no_input("Use enhanced solver? (y/n): ")
    
    # Prompt for debug mode if not specified
    if not args.debug:
        args.debug = get_yes_no_input("Enable debug output? (y/n): ")
    
    # Prompt for time limit if not specified
    if not args.time_limit and not args.analyze_only and not args.mock:
        while True:
            try:
                time_limit_input = input("Enter time limit in seconds (or press Enter for no limit): ")
                if not time_limit_input:
                    break
                args.time_limit = int(time_limit_input)
                if args.time_limit > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Construct paths
    dataset_path = f"datasets/{args.dataset}"
    dataset_name = args.dataset.split('.')[0]
    results_dir = f"results/{dataset_name}"
    
    print(f"\nSolving ITC 2007 Examination Timetabling Problem")
    print(f"Dataset: {args.dataset}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Time limit: {args.time_limit if args.time_limit else 'None'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Solver: {'Enhanced' if args.enhanced else 'Standard'}")
    
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {args.dataset} not found in datasets/ directory")
        return
    
    # Analyze dataset if requested
    if args.analyze_only:
        from utils.solver import analyze_dataset
        analyze_dataset(dataset_path, results_dir)
        return
    
    # Create mock solution if requested
    if args.mock:
        from utils.solver import create_mock_solution
        create_mock_solution(dataset_path, results_dir)
        print(f"Mock solution created in {results_dir}")
        return
    
    # Solve the problem
    start_time = time.time()
    
    try:
        if args.enhanced:
            print("Using enhanced solver")
            from utils.solver_enhanced import solve_itc2007_enhanced
            objective = solve_itc2007_enhanced(dataset_path, results_dir, args.time_limit, args.debug)
        else:
            from utils.solver import solve_itc2007
            objective = solve_itc2007(dataset_path, results_dir, args.time_limit, args.debug)
        
        # Print results
        print(f"Solution found with objective value: {objective}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        print("If this is a Gurobi license error, please ensure you have a valid Gurobi license installed.")
        print("For academic users, free licenses are available at: https://www.gurobi.com/academia/academic-program-and-licenses/")
        print("\nCreating a mock solution instead...")
        from utils.solver import create_mock_solution
        create_mock_solution(dataset_path, results_dir)

if __name__ == "__main__":
    main() 