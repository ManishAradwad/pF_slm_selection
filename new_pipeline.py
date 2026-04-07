"""
New SMS filtering pipeline for pocket-financer.
Focus: actual bank account / credit-debit card transactions only.
Approach: positive filtering (require account + verb) instead of negative (exclude promos).
"""

import re
import os
import pandas as pd

# ── Load & prep ──────────────────────────────────────────────────────────────
sms_db = pd.read_csv("./all_sms.csv")
sms_db['date'] = pd.to_datetime(sms_db['date'], format='ISO8601')

# Strip characters illegal in XML 1.0 (openpyxl/lxml reject them when writing Excel).
# Legal XML 1.0 chars: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
_ILLEGAL_XML_RE = re.compile('[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]')

for col in sms_db.select_dtypes(include=['object', 'string']).columns:
    sms_db[col] = sms_db[col].apply(
        lambda v: _ILLEGAL_XML_RE.sub('', v) if isinstance(v, str) else v
    )

inbox_df = sms_db[sms_db['sender'] != 'me'].copy()

def categorize_sender(sender):
    sender = str(sender).strip()
    if re.match(r'^\+?[0-9]{10,15}$', sender):
        return 'Mobile Number'
    elif re.match(r'^[0-9]{3,9}$', sender):
        return 'Numeric Shortcode'
    elif re.match(r'^[A-Za-z]{2}-.+$', sender):
        return 'Commercial/Brand (Prefixed)'
    elif re.search(r'[A-Za-z]', sender):
        return 'Commercial/Brand (Textual)'
    else:
        return 'Others'

inbox_df['sender_category'] = inbox_df['sender'].apply(categorize_sender)

# Stage 0: Drop personal mobile numbers
commercial_df = inbox_df[inbox_df['sender_category'] != 'Mobile Number'].copy()
print(f"Total commercial/shortcode messages: {len(commercial_df)}")

# ── STAGE 1: AMOUNT REQUIREMENT ──────────────────────────────────────────────
amount_pattern = re.compile(
    r'(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|₹)',
    re.IGNORECASE
)

commercial_df['1_has_amount'] = commercial_df['text'].apply(
    lambda t: bool(amount_pattern.search(t)) if isinstance(t, str) else False
)
s1_count = commercial_df['1_has_amount'].sum()
print(f"\nSTAGE 1 (has amount): {s1_count} retained, {len(commercial_df) - s1_count} dropped")

# ── STAGE 2: REQUIRE MASKED ACCOUNT/CARD NUMBER ─────────────────────────────
# The strongest positive signal — real bank alerts almost always have a masked
# account or card number. This naturally excludes wallets, promos, brokerage, etc.
acct_pattern = re.compile(
    r'a/?c\s*(?:no\.?\s*)?[X*x]+\d+|'          # A/c XX6254, A/C No XXXXXXX7902, a/c *9141
    r'a/?c\s*(?:no\.?\s*)?\*+\d+|'              # A/c **9141
    r'card\s*(?:no\.?\s*)?[Xx*]+\d+|'           # Card x6719, Card XX3782
    r'card\s+\d{4}\b|'                           # Card 0816
    r'card\s+ending\s+[Xx*]*\d+',               # card ending XX3782
    re.IGNORECASE
)

def has_account_or_card(row):
    text = row['text']
    if not isinstance(text, str) or not row['1_has_amount']:
        return False
    return bool(acct_pattern.search(text))

commercial_df['2_has_acct'] = commercial_df.apply(has_account_or_card, axis=1)
s2_count = commercial_df['2_has_acct'].sum()
print(f"STAGE 2 (has account/card): {s2_count} retained, {s1_count - s2_count} dropped")

# ── STAGE 3: REQUIRE TRANSACTION ACTION VERB ─────────────────────────────────
# Must contain a verb indicating an actual completed debit/credit event.
# Filters out promos that mention an account (e.g. "EasyEMI on Card xx6719").
txn_verb_pattern = re.compile(
    r'\b(debited|credited|deducted|spent|paid|received|transferred|sent|reversed|refunded|used|withdrawn)\b|'
    r'\b(money\s+transfer|amt\s+sent|amt\s+received)\b|'
    r"you've\s+hand-?picked",    # OneCard spend notifications
    re.IGNORECASE
)

def has_txn_verb(row):
    text = row['text']
    if not isinstance(text, str) or not row['2_has_acct']:
        return False
    return bool(txn_verb_pattern.search(text))

commercial_df['3_has_verb'] = commercial_df.apply(has_txn_verb, axis=1)
s3_count = commercial_df['3_has_verb'].sum()
print(f"STAGE 3 (has txn verb): {s3_count} retained, {s2_count - s3_count} dropped")

# ── STAGE 4: OTP EXCLUSION (tightened) ───────────────────────────────────────
# Tighter than original: dropped loose 'pin' and 'do not share' which caused
# false positives on legitimate transaction alerts.
otp_pattern = re.compile(
    r'\botp\b|\bone.?time.?password\b|\bverification.?code\b',
    re.IGNORECASE
)

def not_is_otp(row):
    text = row['text']
    if not isinstance(text, str) or not row['3_has_verb']:
        return False
    return not bool(otp_pattern.search(text))

commercial_df['4_not_otp'] = commercial_df.apply(not_is_otp, axis=1)
s4_count = commercial_df['4_not_otp'].sum()
print(f"STAGE 4 (not OTP): {s4_count} retained, {s3_count - s4_count} dropped")

# ── STAGE 5: COLLECT REQUEST EXCLUSION ───────────────────────────────────────
# Remove UPI collect/mandate requests — pending actions, not completed transactions.
# Tighter than original 'request' filter: only matches payment-request phrasing.
collect_pattern = re.compile(
    r'has\s+requested\s+money|'
    r'requested\s+Rs\.?|'
    r'collect\s+request|'
    r'mandate\s+request|'
    r'request\s+from\s+you',
    re.IGNORECASE
)

def not_is_collect(row):
    text = row['text']
    if not isinstance(text, str) or not row['4_not_otp']:
        return False
    return not bool(collect_pattern.search(text))

commercial_df['5_not_collect'] = commercial_df.apply(not_is_collect, axis=1)
s5_count = commercial_df['5_not_collect'].sum()
print(f"STAGE 5 (not collect req): {s5_count} retained, {s4_count - s5_count} dropped")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
os.makedirs('RESULTS/new_pipeline', exist_ok=True)

stages = [
    ('1_has_amount', '1_transactions_split.xlsx'),
    ('2_has_acct', '2_transactions_split.xlsx'),
    ('3_has_verb', '3_transactions_split.xlsx'),
    ('4_not_otp', '4_transactions_split.xlsx'),
    ('5_not_collect', '5_transactions_split.xlsx'),
]

for col, filename in stages:
    path = f'RESULTS/new_pipeline/{filename}'
    lt = commercial_df[commercial_df[col] == True]
    nt = commercial_df[commercial_df[col] == False]
    with pd.ExcelWriter(path) as writer:
        lt.to_excel(writer, sheet_name='likely_transactions', index=False)
        nt.to_excel(writer, sheet_name='non_transactions', index=False)

# ── PRINT SAMPLES FROM FINAL OUTPUT ─────────────────────────────────────────
final = commercial_df[commercial_df['5_not_collect'] == True]

print(f"\n{'='*80}")
print(f"PIPELINE SUMMARY")
print(f"{'='*80}")
print(f"  Commercial/shortcode msgs: {len(commercial_df)}")
print(f"  Stage 1 (has amount):      {s1_count}")
print(f"  Stage 2 (has account/card): {s2_count}")
print(f"  Stage 3 (has txn verb):     {s3_count}")
print(f"  Stage 4 (not OTP):          {s4_count}")
print(f"  Stage 5 (not collect):      {s5_count}")
print(f"{'='*80}")

print(f"\n--- 20 RANDOM SAMPLES FROM FINAL OUTPUT ---")
for text in final['text'].dropna().sample(min(20, len(final)), random_state=42):
    print(f"  * {str(text)[:250].replace(chr(10), ' ')}")
    print()
