import random
import pandas as pd

# Define testing data - in implementation with Fitness Friend will be using our lift database 
exercises = {
    "Legs": ["Squat", "Leg Press", "Lunges", "Leg Curls", "Calf Raise"],
    "Chest": ["Bench Press", "Chest Press", "Push-Ups", "Pec Deck", "Incline Press"],
    "Arms": ["Bicep Curl", "Tricep Extension", "Hammer Curl", "Dumbbell Curl"],
    "Back": [ "Row", "Lat Pulldown", "Pull-Ups", "Cable Row"],
    "Full Body": ["Deadlift", "Clean and Press", "Burpees", "Running"]    # This is all random test data - not corresponding my fitness friend database
}

# Rep ranges based on the goal - low range for strenght with higher weight and high range with hypertrophy
rep_ranges = {
    1: (4, 6), # Strength
    0: (8, 14), # Hypertrophy
}

# Function to generate synthetic data for training takes input in format [_,_,_,_,_,_] with either 0 for no or 1 for yes, for strenght and hypertrophy
def generate_synthetic_data(num_samples):
    data = []
    
    for _ in range(num_samples):
        while True:  # Repeat until valid data is generated
            # Randomly choosing strength or hypertrophy and other body parts being targeted
            goal = random.choice([0, 1])
            legs = random.choice([0, 1])
            chest = random.choice([0, 1])
            arms = random.choice([0, 1])
            back = random.choice([0, 1])
            full_body = random.choice([0, 1])

            # Ensure at least one body part is targeted
            if legs or chest or arms or back or full_body:
                break

        # Set the rep range based on the goal
        rep_min, rep_max = rep_ranges[goal]
        
        # Collecting exercises based on the input - ensuring correct targeting
        lifts = []
        selected_lifts = set()
        
        target_groups = {
            "Legs": legs,
            "Chest": chest,
            "Arms": arms,
            "Back": back,
            "Full Body": full_body
        }
        
        for group, is_targeted in target_groups.items():
            if is_targeted and group in exercises:
                lift = random.choice(exercises[group])
                lifts.append(lift)
                selected_lifts.add(lift)

        # Ensure at least 6 exercises are selected and avoid duplicates
        while 6 < len(lifts) < 9:
            available_groups = [group for group, is_targeted in target_groups.items() if is_targeted and group in exercises]
            if not available_groups:
                break
            group_choice = random.choice(available_groups)
            new_lift = random.choice(exercises[group_choice])
            
            # Avoid duplicates
            if new_lift not in selected_lifts:
                lifts.append(new_lift)
                selected_lifts.add(new_lift)

        # Reps for the exercises
        reps = random.randint(rep_min, rep_max)
        
        # Create the record for this sample (input and output)
        record = {
            "Goal": goal,
            "Legs": legs,
            "Chest": chest,
            "Arms": arms,
            "Back": back,
            "Full Body": full_body,
            "Lifts": ";".join(lifts),
            "Reps": reps
        }
        
        data.append(record)

    # Convert to dataframe and create csv file
    df = pd.DataFrame(data)
    df.to_csv("../data/Synthetic_data_FF.csv", index=False)
    print(f"Generated {len(data)} samples of synthetic data.")

# Running script
synthetic_data = generate_synthetic_data(1000)

