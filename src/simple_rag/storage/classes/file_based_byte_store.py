import os
from langchain.storage import InMemoryByteStore
import logging


class FileBasedByteStore(InMemoryByteStore):
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            self.load()

    def save(self):
        """Save all documents in the store to the local directory."""
        for key, value in self.store.items():
            filepath = os.path.join(self.directory, f"{key}.bin")
            try:
                with open(filepath, "wb") as f:
                    f.write(value)
                    # json.dump({"page_content": value.page_content, "metadata": value.metadata}, f)
            except Exception as e:
                logging.error(f"Error saving document {key}: {e}")

    def load(self):
        """Load all documents from the local directory into the store."""
        for filename in os.listdir(self.directory):
            if filename.endswith(".bin"):
                try:
                    filepath = os.path.join(self.directory, filename)
                    with open(filepath, "rb") as f:
                        data = f.read()
                        key = filename[:-4]  # Remove the .bin extension
                        self.store[key] = data
                except Exception as e:
                    logging.error(f"Error loading file {filename}: {e}")

    def mset(self, key_value_pairs):
        """Override mset to auto-save after updating the store."""
        super().mset(key_value_pairs)
        self.save()
