"""
Go games downloaded from the github repo by: yenw
link: https://github.com/yenw/computer-go-dataset

"""

import os
import zipfile
import shutil


if __name__ == "__main__":
    # we will be working with only the professional go games
    working_dir = "go_games"
    path = "computer-go-dataset/Professional"

    os.makedirs(working_dir, exist_ok=True)

    all_files = os.listdir(path)

    for file_name in all_files:
        if file_name.endswith(".zip"):
            src = os.path.join(path, file_name)
            dst = os.path.join(working_dir, file_name)

            shutil.copy(src, dst)

            with zipfile.ZipFile(dst, "r") as zip_ref:
                zip_ref.extractall(working_dir)

            os.remove(dst)
