# Copyright (c) 2025 Song Huang, Guangxin Jiang, Ying Zhong.
# Licensed under the MIT license.

import os


def generate_alternatives_information_file(L1, L2):
    """
    A function generates the parameters of the Three-Stage Buffer Allocation Problem in a .txt.
    """
    n = 0  # L1 counter for the number of combinations

    # Get the directory where the current script is located.
    # __file__ is a special variable that holds the path to the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Calculate the total number of combinations for the filename.
    # Using integer division // to ensure the result is an integer.
    total_count = (L1 - 1) * (L1 - 2) // 2 * (L2 - 1)

    # Create the dynamic filename.
    filename = f"TPMax_{total_count}.txt"

    # Define the output file path to be in the same directory as the script.
    file_path = os.path.join(script_dir, filename)

    try:
        # 'with open' handles the file closing automatically, even if errors occur.
        with open(file_path, 'w') as writer:
            # Loop to generate data combinations.
            for b1 in range(1, L1 - 1):
                for b2 in range(1, L1 - b1):
                    b3 = L1 - b1 - b2
                    # Ensure b3 is at least 1.
                    if b3 < 1:
                        continue

                    for s1 in range(1, L2):
                        s2 = L2 - s1

                        n += 1  # Increment the counter

                        # Use an f-string for easy and readable string formatting.
                        line = f"{n} {b1} {b2} {b3} {s1} {s2}\n"
                        writer.write(line)

        print(f"Data writing is complete. File saved to: {file_path}")

    except IOError as e:
        # Catches errors related to file input/output.
        print(f"Error while writing file: {e}")


# This block ensures the function runs when the script is executed directly.
if __name__ == "__main__":
    L1 = 50
    L2 = 50
    generate_alternatives_information_file(L1, L2)
