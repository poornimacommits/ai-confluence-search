import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
        self.CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")


settings = Settings()
