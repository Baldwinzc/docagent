"""citelocal_agent package."""

from dotenv import find_dotenv, load_dotenv

# Load .env as early as possible: several modules read os.environ at import
# time (e.g. configuration defaults), so the .env must be loaded before any
# submodule is imported.
load_dotenv(find_dotenv(usecwd=True))

version = "0.1.0"
