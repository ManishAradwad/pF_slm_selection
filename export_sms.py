"""
Export all SMS from sms.db → all_sms.csv + all_sms.json

Handles newer iOS versions (≈iOS 17+) that store message text in the
`attributedBody` BLOB (NSMutableAttributedString typedstream) instead of
the plain `text` column.

Usage:
    python export_sms.py                     # uses ./sms.db, writes ./all_sms.csv + ./all_sms.json
    python export_sms.py path/to/sms.db      # custom input path
"""

import sqlite3
import csv
import json
import sys
from datetime import datetime, timedelta

APPLE_EPOCH = datetime(2001, 1, 1)


def decode_attributed_body(blob: bytes) -> str | None:
    """Extract plain text from an NSMutableAttributedString typedstream blob.

    iOS stores the SMS body as a serialised NSMutableAttributedString when the
    plain `text` column is NULL.  The binary layout (Apple typedstream) embeds
    the raw UTF-8 string after the NSString class marker with a variable-length
    integer prefix:
        <0x80  → single-byte length
        0x81   → next 2 bytes, little-endian uint16
        0x82   → next 4 bytes, little-endian uint32
    """
    if not blob:
        return None

    idx = blob.find(b"NSString")
    if idx == -1:
        return None

    remaining = blob[idx + len(b"NSString") :]

    # The string data follows a \x01+ marker after the class hierarchy
    plus_idx = remaining.find(b"\x01+")
    if plus_idx == -1:
        return None

    pos = plus_idx + 2  # skip past \x01+

    # Variable-length integer
    length_byte = remaining[pos]
    pos += 1

    if length_byte < 0x80:
        text_length = length_byte
    elif length_byte == 0x81:
        text_length = remaining[pos] | (remaining[pos + 1] << 8)
        pos += 2
    elif length_byte == 0x82:
        text_length = (
            remaining[pos]
            | (remaining[pos + 1] << 8)
            | (remaining[pos + 2] << 16)
            | (remaining[pos + 3] << 24)
        )
        pos += 4
    else:
        return None

    text_bytes = remaining[pos : pos + text_length]
    try:
        return text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return text_bytes.decode("utf-8", errors="replace")


def apple_ns_to_datetime(ns: int) -> datetime:
    """Convert Apple-epoch nanosecond timestamp to a Python datetime."""
    return APPLE_EPOCH + timedelta(seconds=ns / 1e9)


def export(db_path: str = "sms.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT m.ROWID,
               m.date,
               CASE WHEN m.is_from_me = 1 THEN 'me' ELSE h.id END AS sender,
               m.text,
               m.attributedBody,
               m.is_from_me,
               m.service
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.date > 0
        ORDER BY m.date
        """
    )

    rows = []
    stats = {"total": 0, "from_text": 0, "from_blob": 0, "no_content": 0}

    for rowid, date_ns, sender, text, blob, is_from_me, service in cursor:
        stats["total"] += 1

        # Prefer the plain text column; fall back to attributedBody blob
        body = text
        if not body or not body.strip():
            body = decode_attributed_body(blob)
            if body:
                stats["from_blob"] += 1
            else:
                stats["no_content"] += 1
                continue  # skip messages with no recoverable text
        else:
            stats["from_text"] += 1

        dt = apple_ns_to_datetime(date_ns)

        rows.append(
            {
                "id": rowid,
                "date": dt.isoformat(),
                "sender": sender or "",
                "text": body,
                "is_from_me": bool(is_from_me),
                "service": service or "",
            }
        )

    conn.close()

    # Write CSV
    csv_path = "all_sms.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "date", "sender", "text", "is_from_me", "service"])
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    json_path = "all_sms.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(rows)} messages → {csv_path}, {json_path}")
    print(f"  Text column:        {stats['from_text']}")
    print(f"  attributedBody:     {stats['from_blob']}")
    print(f"  No content (skip):  {stats['no_content']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sms.db"
    export(path)
