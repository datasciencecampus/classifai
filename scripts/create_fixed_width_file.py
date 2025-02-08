"""
Generate a synthetic input file.

Example usage:
python scripts/create_fixed_width_file.py -n 100 data/soc_fixed_wd_100.in
"""

import random


def create_fixed_width_file(filename, num_rows):
    """Create a fixed width SOC file.

    Arguments
    ---------
    filename (str): The filename to write to
    num_rows (int): Number of rows to write
    """
    # List of sample occupations
    occupations = [
        "ESTATE AGENT",
        "TEACHER",
        "DOCTOR",
        "SOFTWARE ENGINEER",
        "CHEF",
        "POLICE OFFICER",
        "NURSE",
        "ACCOUNTANT",
        "LAWYER",
        "ELECTRICIAN",
        "PLUMBER",
        "ARCHITECT",
        "DENTIST",
        "MECHANIC",
        "PILOT",
        "PHOTOGRAPHER",
    ]

    # Create set to track used IDs
    used_ids = set()

    with open(filename, "w") as f:
        for _ in range(num_rows):
            # Generate unique 7-digit ID
            while True:
                id_num = random.randint(1000000, 9999999)
                if id_num not in used_ids:
                    used_ids.add(id_num)
                    break

            # Select random occupation
            occupation = random.choice(occupations)

            # Ensure occupation doesn't exceed 42 characters
            occupation = occupation[:42]

            # Write formatted line to file
            f.write(f"{id_num}{occupation}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a fixed-width file with IDs and occupations"
    )
    parser.add_argument(
        "-n", type=int, required=True, help="Number of rows to generate"
    )
    parser.add_argument("filename", help="Output filename")

    args = parser.parse_args()

    create_fixed_width_file(args.filename, args.n)
    print(f"Created file '{args.filename}' with {args.n} rows.")
