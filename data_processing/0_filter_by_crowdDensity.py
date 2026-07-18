import argparse
import os
import pandas as pd

def filter_by_crowd_density(input_file, output_file, crowd_density):
    """
    Read a CSV file, filter rows by crowdDensity value,
    and save the result to a new CSV file.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
        crowd_density: Target crowdDensity value to keep.
    """
    try:
        # Read the CSV file.
        df = pd.read_csv(input_file)

        # Check if the crowdDensity column exists.
        if 'crowdDensity' not in df.columns:
            print("Error: 'crowdDensity' column does not exist in the CSV file.")
            return

        # Filter rows where crowdDensity matches the target value.
        filtered_df = df[df['crowdDensity'] == crowd_density]

        # Save the filtered result to a new CSV file.
        filtered_df.to_csv(output_file, index=False)

        print(f"Filtered {len(filtered_df)} rows for crowdDensity='{crowd_density}'. Saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


def build_output_csv_path(input_csv, output_csv, crowd_density):
    if output_csv:
        return output_csv
    input_base, input_ext = os.path.splitext(input_csv)
    return f"{input_base}_{crowd_density}{input_ext}"


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows by crowdDensity.")
    parser.add_argument("--input_csv", type=str, default="SpatialVID_HQ_metadata.csv", help="Input CSV path.")
    parser.add_argument("--crowd_density", type=str, default="Sparse", help="Target crowdDensity value to keep.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <input_csv>_<crowd_density>.csv",
    )
    args = parser.parse_args()

    output_csv = build_output_csv_path(args.input_csv, args.output_csv, args.crowd_density)
    filter_by_crowd_density(args.input_csv, output_csv, args.crowd_density)


if __name__ == "__main__":
    main()
