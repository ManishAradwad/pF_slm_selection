import os
import shutil
import plistlib

# Path to your iTunes backup folder
backup_base = os.path.expandvars(
    r"./"
)

# Find the latest backup
backups = sorted(
    [os.path.join(backup_base, d) for d in os.listdir(backup_base)],
    key=os.path.getmtime,
    reverse=True,
)
latest_backup = backups[0]
print(f"Latest backup: {latest_backup}")

# The SMS database has this known manifest hash
# You can also search the Manifest.db for 'sms.db'
manifest_db = os.path.join(latest_backup, "Manifest.db")

import sqlite3
conn = sqlite3.connect(manifest_db)
cursor = conn.cursor()
cursor.execute(
    "SELECT fileID, relativePath FROM Files WHERE relativePath LIKE '%sms.db'"
)
for file_id, rel_path in cursor.fetchall():
    print(f"Found: {rel_path}")
    # The actual file is stored as <first2chars_of_hash>/<full_hash>
    src = os.path.join(latest_backup, file_id[:2], file_id)
    dst = "sms.db"
    shutil.copy2(src, dst)
    print(f"Copied to: {os.path.abspath(dst)}")

conn.close()