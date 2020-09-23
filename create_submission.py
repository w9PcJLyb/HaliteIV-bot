import re
import os
import argparse
from glob import glob


def write_file(file_name, out_file):
    out_file.write("#" * 40 + "\n")
    out_file.write(
        "### " + os.path.splitext(os.path.basename(file_name))[0].upper() + "\n"
    )
    out_file.write("#" * 40 + "\n")

    with open(file_name, "r") as file:
        for line in file.readlines():
            if re.match("from \..* import .*|from halite import .*", line):
                line = f"# {line}"

            if line.startswith("IS_KAGGLE = False"):
                line = "IS_KAGGLE = True\n"

            if line.startswith("LEVEL = logging."):
                line = "LEVEL = logging.INFO\n"

            out_file.write(line)

    out_file.write("\n\n\n")


def main(submission_name):
    with open(submission_name, "w") as out_file:
        write_file("halite/logger.py", out_file)
        write_file("halite/board.py", out_file)

        for file_name in glob("halite/*.py"):
            if (
                file_name.endswith("__init__.py")
                or file_name.endswith("board.py")
                or file_name.endswith("logger.py")
            ):
                continue

            write_file(file_name, out_file)

        write_file("agent.py", out_file)
    print(submission_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="submission name")
    flags = parser.parse_args()
    main(flags.name)
