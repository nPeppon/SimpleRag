import os


def get_data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


def get_log_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "log")


# Ensure the directories exist
os.makedirs(get_log_dir(), exist_ok=True)
os.makedirs(get_data_dir(), exist_ok=True)


if __name__ == "__main__":
    print(get_data_dir())
    if os.path.exists(get_data_dir()):
        print("Data directory exists.")
