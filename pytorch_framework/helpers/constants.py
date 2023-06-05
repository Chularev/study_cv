from pathlib import Path


def get_project_root_dir():
    return str( Path(__file__).absolute().parent.parent )

if __name__ == "__main__":
    print('root = ', get_project_root_dir())