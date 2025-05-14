import os

# Step 1: Set environment variables
os.environ['TMPDIR'] = '/opt/tmp'
os.environ['TMP'] = '/opt/tmp'
os.environ['TEMP'] = '/opt/tmp'

# Step 3: Ensure directory exists
os.makedirs('/opt/tmp', exist_ok=True)
os.chmod('/opt/tmp', 0o1777)  # Adjust permissions for security

# Step 2: Configure tempfile module
import tempfile
tempfile.tempdir = '/opt/tmp'
