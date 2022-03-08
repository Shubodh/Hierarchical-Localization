from shutil import copy2
import argparse
from pathlib import Path


def sampling_10(input_path, output_path):
    input_path = Path(input_path)
    assert input_path.exists(), input_path
    all_files = list(input_path.glob('*'))
    for every_file in all_files:
        if "camera.yaml" in str(every_file):
            print(every_file, output_path)
            copy2(every_file, output_path)
        else:
            end_str = str(every_file.name)
            end_int = int(''.join(filter(str.isdigit, end_str)))
            if end_int % 10 == 0:
                print(every_file, output_path)
                copy2(every_file, output_path)


if __name__ == "__main__":
    # main_dummy()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    sampling_10(**args.__dict__)
