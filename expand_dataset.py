"""
Expand extraction_ds.jsonl with 100 new samples (50 txn + 50 non-txn) from 2025/2026.
"""

import pandas as pd
import json
import re

FLAGS = re.DOTALL | re.IGNORECASE

MONTH_MAP = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
}

def normalize_date(raw: str) -> str:
    raw = raw.strip().rstrip('.')
    # YYYY-MM-DD
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # DD/MM/YY
    m = re.match(r'(\d{1,2})/(\d{2})/(\d{2})$', raw)
    if m:
        return f"{int(m.group(1)):02d}-{m.group(2)}-20{m.group(3)}"
    # DD/MMM/YYYY or with _value suffix
    m = re.match(r'(\d{1,2})/([A-Za-z]{3})/(\d{4})', raw)
    if m:
        return f"{int(m.group(1)):02d}-{MONTH_MAP[m.group(2).lower()]}-{m.group(3)}"
    # DD-Mon-YY or DD-MON-YY
    m = re.match(r'(\d{1,2})-([A-Za-z]{3})-(\d{2,4})', raw)
    if m:
        yr = m.group(3) if len(m.group(3)) == 4 else f"20{m.group(3)}"
        return f"{int(m.group(1)):02d}-{MONTH_MAP[m.group(2).lower()]}-{yr}"
    # DDMonYY (SBI: 21Jan26)
    m = re.match(r'(\d{1,2})([A-Za-z]{3})(\d{2})$', raw)
    if m:
        return f"{int(m.group(1)):02d}-{MONTH_MAP[m.group(2).lower()]}-20{m.group(3)}"
    # DD-MM-YY
    m = re.match(r'(\d{2})-(\d{2})-(\d{2})$', raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-20{m.group(3)}"
    return raw

def parse_amount(raw: str) -> float:
    return float(raw.replace(',', ''))

# ── Extractors (all use re.DOTALL so newlines in SMS don't break matching) ──

def extract_hdfc_sent(text):
    m = re.search(r'Sent\s+Rs\.?([\d,]+(?:\.\d+)?)\s+From\s+HDFC\s+Bank\s+(A/C\s*[\*x]?\d+)\s+To\s+(.+?)\s+On\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(3).strip(), "date": normalize_date(m.group(4)), "type": "debit", "account": m.group(2).strip()}

def extract_hdfc_txn_card(text):
    m = re.search(r'Txn\s+Rs\.?([\d,]+(?:\.\d+)?)\s+On\s+HDFC\s+Bank\s+(Card\s+\d+)\s+At\s+(.+?)\s+by\s+UPI\s+\S+\s+On\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(3).strip(), "date": normalize_date(m.group(4)), "type": "debit", "account": m.group(2).strip()}

def extract_hdfc_spent_on(text):
    m = re.search(r'Spent\s+Rs\.?([\d,]+(?:\.\d+)?)\s+On\s+HDFC\s+Bank\s+(Card\s+\d+)\s+At\s+(.+?)\s+On\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(3).strip(), "date": normalize_date(m.group(4)), "type": "debit", "account": m.group(2).strip()}

def extract_hdfc_rs_spent(text):
    m = re.search(r'Rs\.?([\d,]+(?:\.\d+)?)\s+spent\s+on\s+HDFC\s+Bank\s+(Card\s+x?\d+)\s+at\s+(.+?)\s+on\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(3).strip(), "date": normalize_date(m.group(4)), "type": "debit", "account": m.group(2).strip()}

def extract_hdfc_credit_alert(text):
    m = re.search(r'Credit\s+Alert!\s+Rs\.?([\d,]+(?:\.\d+)?)\s+credited\s+to\s+HDFC\s+Bank\s+(A/c\s*[Xx]*\d+)\s+on\s+(\S+)\s+from\s+VPA\s+(\S+)', text, FLAGS)
    if not m: return None
    vpa = re.sub(r'\s*\(UPI.*', '', m.group(4).strip().rstrip(')'))
    return {"amount": parse_amount(m.group(1)), "merchant": vpa, "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(2).strip()}

def extract_hdfc_card_payment(text):
    m = re.search(r'Online\s+Payment\s+of\s+Rs\.?([\d,]+(?:\.\d+)?)\s+.*?credited\s+to\s+your\s+(card\s+ending\s+\d+)\s+On\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": None, "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(2).strip()}

def extract_hdfc_neft_deposit(text):
    m = re.search(r'INR\s+([\d,]+(?:\.\d+)?)\s+deposited\s+in\s+HDFC\s+Bank\s+(A/c\s*[Xx]*\d+)\s+on\s+(\S+)\s+for\s+NEFT\s+Cr-(.+?)Avl', text, FLAGS)
    if not m: return None
    parts = m.group(4).split('-')
    merchant = parts[1].strip() if len(parts) > 1 else None
    return {"amount": parse_amount(m.group(1)), "merchant": merchant, "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(2).strip()}

def extract_hdfc_refund(text):
    m = re.search(r'Rs\.?\s*([\d,]+(?:\.\d+)?)\s+refunded\s+by\s+(.+?)\s+on\s+(\S+).*?(?:HDFC\s+Bank\s+)?(Credit\s+Card\s+\d+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(2).strip(), "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(4).strip()}

def extract_icici_spent(text):
    m = re.search(r'INR\s+([\d,]+(?:\.\d+)?)\s+spent\s+using\s+ICICI\s+Bank\s+(Card\s+XX\d+)\s+on\s+(\S+)\s+on\s+(.+?)\.\s*Avl', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": m.group(4).strip(), "date": normalize_date(m.group(3)), "type": "debit", "account": m.group(2).strip()}

def extract_icici_debited(text):
    m = re.search(r'ICICI\s+Bank\s+(Credit\s+Card\s+XX\d+)\s+debited\s+for\s+INR\s+([\d,]+(?:\.\d+)?)\s+on\s+(\S+)\s+for\s+UPI-\d+-(.+?)\.?\s*(?:To\s+dispute|$)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(2)), "merchant": m.group(4).strip().rstrip('.'), "date": normalize_date(m.group(3)), "type": "debit", "account": m.group(1).strip()}

def extract_icici_payment(text):
    m = re.search(r'Payment\s+of\s+Rs\s+([\d,]+(?:\.\d+)?)\s+has\s+been\s+received\s+on\s+your\s+ICICI\s+Bank\s+(Credit\s+Card\s+XX\d+)\s+through\s+.+?\s+on\s+(\S+)', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(1)), "merchant": None, "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(2).strip()}

def extract_sbi_credit(text):
    m = re.search(r'(A/c\s*X\d+)-credited\s+by\s+Rs\.?([\d,]+(?:\.\d+)?)\s+on\s+(\S+)\s+transfer\s+from\s+(.+?)\s+Ref', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(2)), "merchant": m.group(4).strip(), "date": normalize_date(m.group(3)), "type": "credit", "account": m.group(1).strip()}

def extract_sbi_debit(text):
    m = re.search(r'(A/c\s*X\d+)-debited\s+by\s+Rs\.?([\d,]+(?:\.\d+)?)\s+on\s+(\S+)\s+transfer\s+to\s+(.+?)\s+Ref', text, FLAGS)
    if not m: return None
    return {"amount": parse_amount(m.group(2)), "merchant": m.group(4).strip(), "date": normalize_date(m.group(3)), "type": "debit", "account": m.group(1).strip()}

EXTRACTORS = [
    extract_hdfc_sent, extract_hdfc_txn_card, extract_hdfc_spent_on,
    extract_hdfc_rs_spent, extract_hdfc_credit_alert, extract_hdfc_card_payment,
    extract_hdfc_neft_deposit, extract_hdfc_refund,
    extract_icici_spent, extract_icici_debited, extract_icici_payment,
    extract_sbi_credit, extract_sbi_debit,
]

def extract_transaction(text):
    for fn in EXTRACTORS:
        r = fn(text)
        if r is not None:
            return r
    return None

# ── Load data ────────────────────────────────────────────────────────────────
existing_texts = set()
with open('DATA/extraction_ds.jsonl') as f:
    for line in f:
        existing_texts.add(json.loads(line)['sms'][:80])

txn_df = pd.read_excel('RESULTS/new_pipeline/5_transactions_split.xlsx', sheet_name='likely_transactions')
txn_df['date'] = pd.to_datetime(txn_df['date'])
non_df = pd.read_excel('RESULTS/new_pipeline/5_transactions_split.xlsx', sheet_name='non_transactions')
non_df['date'] = pd.to_datetime(non_df['date'])

# Filter false positives from txn sheet (statement reminders, CKYC)
fp_pat = re.compile(r'Statement is sent|is due by|CKYC|Pay Total Amount Due', re.IGNORECASE)
txn_df = txn_df[~txn_df['text'].apply(lambda t: bool(fp_pat.search(str(t))))]

txn_25 = txn_df[txn_df['date'].dt.year == 2025].copy()
txn_26 = txn_df[txn_df['date'].dt.year == 2026].copy()
non_25 = non_df[non_df['date'].dt.year == 2025].copy()
non_26 = non_df[non_df['date'].dt.year == 2026].copy()

def not_dup(text):
    return str(text)[:80] not in existing_texts

# ── Categorize transactions ───────────────────────────────────────────────────
def cat_txn(text):
    t = str(text)
    if re.search(r'Sent\s+Rs', t, FLAGS): return 'hdfc_sent'
    if re.search(r'Txn\s+Rs.*On\s+HDFC\s+Bank\s+Card', t, FLAGS): return 'hdfc_txn'
    if re.search(r'Spent\s+Rs.*On\s+HDFC\s+Bank\s+Card', t, FLAGS): return 'hdfc_spent'
    if re.search(r'Rs.*spent\s+on\s+HDFC\s+Bank\s+Card', t, FLAGS): return 'hdfc_rs_spent'
    if re.search(r'Credit\s+Alert', t, FLAGS): return 'hdfc_credit_alert'
    if re.search(r'Online\s+Payment.*card\s+ending', t, FLAGS): return 'hdfc_card_payment'
    if re.search(r'deposited.*NEFT', t, FLAGS): return 'hdfc_neft'
    if re.search(r'refunded\s+by', t, FLAGS): return 'hdfc_refund'
    if re.search(r'spent\s+using\s+ICICI\s+Bank\s+Card', t, FLAGS): return 'icici_spent'
    if re.search(r'ICICI.*debited\s+for\s+INR', t, FLAGS): return 'icici_debited'
    if re.search(r'Payment.*received.*ICICI', t, FLAGS): return 'icici_payment'
    if re.search(r'A/c\s*X\d+-credited', t, FLAGS): return 'sbi_credit'
    if re.search(r'A/c\s*X\d+-debited', t, FLAGS): return 'sbi_debit'
    return 'other'

for df in [txn_25, txn_26]:
    df['cat'] = df['text'].apply(cat_txn)

# ── Select 50 transactions: ~30 from 2026, ~20 from 2025 ───────────────────
targets_26 = {'hdfc_sent': 8, 'hdfc_txn': 7, 'hdfc_spent': 3, 'hdfc_credit_alert': 3,
              'hdfc_card_payment': 2, 'hdfc_neft': 2, 'icici_spent': 3,
              'icici_debited': 5, 'icici_payment': 2, 'sbi_credit': 2}

targets_25 = {'hdfc_sent': 4, 'hdfc_txn': 4, 'hdfc_spent': 1, 'hdfc_rs_spent': 2,
              'hdfc_credit_alert': 2, 'hdfc_card_payment': 1, 'hdfc_neft': 1,
              'hdfc_refund': 1, 'icici_spent': 2, 'icici_payment': 1, 'sbi_credit': 1}

selected_txns = []
failed = []

def pick_txns(df, targets, year_label):
    for cat, n in targets.items():
        pool = df[(df['cat'] == cat) & df['text'].apply(not_dup)]
        if pool.empty:
            print(f"  WARN: no {cat} in {year_label}")
            continue
        for _, row in pool.sample(min(n, len(pool)), random_state=42).iterrows():
            text = str(row['text'])
            result = extract_transaction(text)
            if result is None:
                failed.append((year_label, cat, text[:150]))
            else:
                selected_txns.append({'sms': text, 'sender': str(row['sender']), 'expected': json.dumps(result)})

pick_txns(txn_26, targets_26, '2026')
pick_txns(txn_25, targets_25, '2025')

print(f"Transactions selected: {len(selected_txns)}")
if failed:
    print(f"Failed extractions ({len(failed)}):")
    for y, c, t in failed:
        print(f"  [{y}][{c}] {t}")

# ── Categorize non-transactions ───────────────────────────────────────────────
def cat_non(text):
    t = str(text)
    if re.search(r'\bOTP\b|\bone.time.pass', t, re.IGNORECASE): return 'otp'
    if re.search(r'Statement\s+is\s+sent|Amount\s+Due|minimum.*due\s+by', t, re.IGNORECASE): return 'statement'
    if re.search(r'PAYMENT.*RECEIVED.*CREDIT\s+CARD|AVAILABLE\s+LIMIT.*CREDIT', t, re.IGNORECASE): return 'hdfc_pay_confirm'
    if re.search(r'CabID|MoveInSync|within\s+\d+km.*pickup', t, re.IGNORECASE): return 'cab'
    if re.search(r'SIP\s+installment|Mutual\s+Fund|NAV\s+of|units\s+are\s+allotted', t, re.IGNORECASE): return 'mf_sip'
    if re.search(r'KOTAK\s+SECURITIES|Securities\s+bal|NSDL.*blocked\s+for\s+debit|NSE.*reported', t, re.IGNORECASE): return 'stock'
    if re.search(r'split.*bill|axio\.co', t, re.IGNORECASE): return 'bill_split'
    if re.search(r'Simpl|Pay\s+Later.*Simpl|instalment.*Simpl', t, re.IGNORECASE): return 'wallet'
    if re.search(r'Apay\s+[Bb]alance|Amazon\s+Pay\s+balance|Juspay', t, re.IGNORECASE): return 'wallet'
    if re.search(r'Pre.?approved.*[Ll]oan|Personal\s+Loan.*approved|loan.*ready', t, re.IGNORECASE): return 'loan_offer'
    if re.search(r'KYC.*expir|Complete\s+KYC|KYC\s+Link', t, re.IGNORECASE): return 'kyc'
    if re.search(r'order.*could\s+not|DELIVERY\s+ATTEMPTED|dispatch|your\s+order.*track', t, re.IGNORECASE): return 'delivery'
    if re.search(r'Refund.*initiated|refund.*credited\s+by.*Aug|Refund.*Zomato', t, re.IGNORECASE): return 'refund_notif'
    if re.search(r'iMobile.*activated|NetBanking.*verify|CKYC\s+Number|AutoPay.*Success|E-mandate.*Success', t, re.IGNORECASE): return 'banking_misc'
    if re.search(r'ebill|JioHome|Bill\s+Summary|Bill\s+period', t, re.IGNORECASE): return 'utility_bill'
    if re.search(r'[Rr]echarge|EXPIR.*pack|plan.*expir|Services\s+stopped', t, re.IGNORECASE): return 'telecom'
    return 'other'

for df in [non_25, non_26]:
    df['cat'] = df['text'].apply(cat_non)

# ── Select 50 non-transactions: ~25 from 2026, ~25 from 2025 ─────────────────
targets_non_26 = {'otp': 5, 'statement': 2, 'hdfc_pay_confirm': 2, 'cab': 2,
                  'mf_sip': 1, 'stock': 2, 'wallet': 2, 'loan_offer': 1,
                  'banking_misc': 2, 'utility_bill': 1, 'telecom': 2, 'other': 3}

targets_non_25 = {'otp': 4, 'stock': 2, 'bill_split': 2, 'wallet': 2,
                  'loan_offer': 1, 'kyc': 1, 'delivery': 1, 'refund_notif': 1,
                  'banking_misc': 1, 'telecom': 2, 'mf_sip': 1, 'other': 5}

selected_non = []

def pick_non(df, targets, year_label):
    for cat, n in targets.items():
        pool = df[(df['cat'] == cat) & df['text'].apply(not_dup)]
        if pool.empty:
            print(f"  WARN: no {cat} in {year_label} non-txn")
            continue
        for _, row in pool.sample(min(n, len(pool)), random_state=42).iterrows():
            selected_non.append({'sms': str(row['text']), 'sender': str(row['sender']), 'expected': 'null'})

pick_non(non_26, targets_non_26, '2026')
pick_non(non_25, targets_non_25, '2025')

print(f"Non-transactions selected: {len(selected_non)}")
print(f"Total new samples: {len(selected_txns) + len(selected_non)}")

# ── Verify all transaction extractions ───────────────────────────────────────
print("\n=== Sample transaction extractions ===")
for entry in selected_txns[:8]:
    sms_preview = entry['sms'].replace('\n', ' ')[:100]
    print(f"  SMS: {sms_preview}")
    print(f"  GT:  {entry['expected']}")
    print()

# ── Append to dataset ─────────────────────────────────────────────────────────
all_new = selected_txns + selected_non
# Interleave so dataset isn't all txn then all non-txn
import random
random.seed(7)
random.shuffle(all_new)

output_path = 'DATA/extraction_ds.jsonl'
with open(output_path, 'a') as f:
    for entry in all_new:
        row = {'sms': entry['sms'], 'sender': entry['sender'], 'expected': entry['expected']}
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"\nAppended {len(all_new)} entries to {output_path}")
print(f"New total: 100 + {len(all_new)} = {100 + len(all_new)} samples")
