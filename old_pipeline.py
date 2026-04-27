import os
import re
import sys
import pandas as pd

def main():
    sms_db = pd.read_csv("./all_sms.csv")
    sms_db['date'] = pd.to_datetime(sms_db['date'], unit='s')
    
    sms_db = sms_db[sms_db['sender'] != 'me']

    # Check whether `id` is sequential and identify gaps/duplicates
    id_series = pd.to_numeric(sms_db["id"], errors="coerce").dropna().astype(int)

    row_count = len(sms_db)
    min_id = id_series.min()
    max_id = id_series.max()
    unique_ids = id_series.nunique()

    is_exact_1_to_n = (min_id == 1) and (max_id == row_count) and (unique_ids == row_count)
    is_contiguous_min_to_max = (unique_ids == (max_id - min_id + 1))

    print(f"rows: {row_count}")
    print(f"id min/max: {min_id}/{max_id}")
    print(f"unique ids: {unique_ids}")
    print(f"sequential 1..rows? {is_exact_1_to_n}")
    print(f"contiguous min..max? {is_contiguous_min_to_max}")

    # Missing IDs in the observed min..max range
    expected_ids = pd.Index(range(min_id, max_id + 1))
    missing_ids = expected_ids.difference(pd.Index(id_series.unique()))

    print(f"missing id count: {len(missing_ids)}")
    print("first 20 missing ids:", missing_ids[:20].tolist())

    # Duplicate IDs (if any)
    dup_counts = id_series.value_counts()
    duplicate_ids = dup_counts[dup_counts > 1]
    print(f"duplicate id count: {len(duplicate_ids)}")
    if len(duplicate_ids) > 0:
        print("first 20 duplicate ids:", duplicate_ids.head(20).to_dict())

    # Filter out sent messages
    inbox_df = sms_db[sms_db['sender'] != 'me'].copy()

    # Basic sender stats
    unique_senders = inbox_df['sender'].nunique()
    print(f"Total incoming messages: {len(inbox_df)}")
    print(f"Total unique senders: {unique_senders}")

    # Top 20 senders by volume
    top_senders = inbox_df['sender'].value_counts().head(50)
    print("\nTop 20 senders by message volume:")
    print(top_senders)

    def categorize_sender(sender):
        sender = str(sender).strip()
        
        # 1. Regular Mobile Number: optional '+' followed by 10 to 15 digits
        if re.match(r'^\+?[0-9]{10,15}$', sender):
            return 'Mobile Number'
        
        # 2. Numeric Shortcode: purely numeric, less than 10 digits
        elif re.match(r'^[0-9]{3,9}$', sender):
            return 'Numeric Shortcode'
        
        # 3. Commercial/Brand: standard XX-YYYYYY Indian gateway format
        elif re.match(r'^[A-Za-z]{2}-.+$', sender):
            return 'Commercial/Brand (Prefixed)'
        
        # 4. Commercial/Brand: other textual names (e.g., 'JioPay', 'HDFCBK' without prefix)
        elif re.search(r'[A-Za-z]', sender):
            return 'Commercial/Brand (Textual)'
        
        # 5. Others: to catch edge cases
        else:
            return 'Others'

    # Apply the categorization
    inbox_df['sender_category'] = inbox_df['sender'].apply(categorize_sender)

    # Get counts
    category_counts = inbox_df['sender_category'].value_counts()
    print("\nMessage volume by Sender Category:")
    print(category_counts, "\n")

    # Let's inspect unique senders in other/edge-case categories
    others_senders = inbox_df[inbox_df['sender_category'] == 'Others']['sender'].unique()
    print(f"Unique senders in 'Others': {len(others_senders)}")
    if len(others_senders) > 0:
        print(others_senders[:20])

    # 1. Focus only on Commercial and Shortcode messages (Drop personal mobile numbers)
    commercial_df = inbox_df[inbox_df['sender_category'] != 'Mobile Number'].copy()

    # ==========================================
    # STAGE 1: AMOUNT REQUIREMENT 
    # ==========================================
    # Transactions almost strictly require a currency format + number. 
    # We're making sure we capture varied spaces between currency like 'Rs. 500', 'Rs.500', 'INR500', '₹ 100'.
    amount_pattern = re.compile(
        r'(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|₹)', 
        re.IGNORECASE
    )

    def has_amount(text):
        if not isinstance(text, str):
            return False
        return bool(amount_pattern.search(text))

    print("\nSTAGE 1: Applying amount requirement filter...")
    commercial_df['1_has_amount'] = commercial_df['text'].apply(has_amount)

    # Analyze breakdown
    txn_counts_1 = commercial_df['1_has_amount'].value_counts()
    print(f"Total Commercial/Shortcode messages: {len(commercial_df)}")
    print("Messages with Amount vs Without (Likely Transaction vs Not):")
    print(txn_counts_1)
    print(f"Percentage retaining amount: {(txn_counts_1.get(True, 0) / len(commercial_df)) * 100:.2f}%\n")

    os.makedirs('RESULTS', exist_ok=True)

    # Separate the datasets based on Stage 1
    likely_transactions = commercial_df[commercial_df['1_has_amount'] == True]
    non_transactions = commercial_df[commercial_df['1_has_amount'] == False]

    output_path = 'RESULTS/1_transactions_split.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        likely_transactions.to_excel(writer, sheet_name='likely_transactions', index=False)
        non_transactions.to_excel(writer, sheet_name='non_transactions', index=False)

    print(f"Saved {len(likely_transactions)} likely transactions and {len(non_transactions)} non-transactions to {output_path}")

    # ==========================================
    # STAGE 2: REQUIRE MASKED ACCOUNT/CARD NUMBER
    # ==========================================
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
        if not isinstance(text, str):
            return False
        if not row['1_has_amount']:
            return False
        return bool(acct_pattern.search(text))

    print('\nSTAGE 2: Applying masked account/card number requirement...')
    commercial_df['2_has_acct'] = commercial_df.apply(has_account_or_card, axis=1)

    txn_counts_2 = commercial_df['2_has_acct'].value_counts()
    print(f'Messages with account/card number vs without:')
    print(txn_counts_2)
    print(f'Percentage retained: {(txn_counts_2.get(True, 0) / len(commercial_df)) * 100:.2f}%\n')

    likely_transactions = commercial_df[commercial_df['2_has_acct'] == True]
    non_transactions = commercial_df[commercial_df['2_has_acct'] == False]

    output_path = 'RESULTS/2_transactions_split.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        likely_transactions.to_excel(writer, sheet_name='likely_transactions', index=False)
        non_transactions.to_excel(writer, sheet_name='non_transactions', index=False)

    print(f'Saved {len(likely_transactions)} likely transactions and {len(non_transactions)} non-transactions to {output_path}')

    # ==========================================
    # STAGE 3: REQUIRE TRANSACTION ACTION VERB
    # ==========================================
    txn_verb_pattern = re.compile(
        r'\b(debited|credited|deducted|spent|paid|received|transferred|sent|reversed|refunded|used)\b|'
        r'\b(money\s+transfer|amt\s+sent|amt\s+received)\b',
        re.IGNORECASE
    )

    def has_txn_verb(row):
        text = row['text']
        if not isinstance(text, str):
            return False
        if not row['2_has_acct']:
            return False
        return bool(txn_verb_pattern.search(text))

    print('\nSTAGE 3: Applying transaction action verb requirement...')
    commercial_df['3_has_verb'] = commercial_df.apply(has_txn_verb, axis=1)

    txn_counts_3 = commercial_df['3_has_verb'].value_counts()
    print(f'Messages with transaction verb vs without:')
    print(txn_counts_3)
    print(f'Percentage retained: {(txn_counts_3.get(True, 0) / len(commercial_df)) * 100:.2f}%\n')

    likely_transactions = commercial_df[commercial_df['3_has_verb'] == True]
    non_transactions = commercial_df[commercial_df['3_has_verb'] == False]

    output_path = 'RESULTS/3_transactions_split.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        likely_transactions.to_excel(writer, sheet_name='likely_transactions', index=False)
        non_transactions.to_excel(writer, sheet_name='non_transactions', index=False)

    print(f'Saved {len(likely_transactions)} likely transactions and {len(non_transactions)} non-transactions to {output_path}')

    # ==========================================
    # STAGE 4: OTP EXCLUSION (tightened)
    # ==========================================
    otp_pattern = re.compile(
        r'\botp\b|\bone.?time.?password\b|\bverification.?code\b',
        re.IGNORECASE
    )

    def not_is_otp(row):
        text = row['text']
        if not isinstance(text, str):
            return False
        if not row['3_has_verb']:
            return False
        return not bool(otp_pattern.search(text))

    print('\nSTAGE 4: Applying OTP exclusion filter...')
    commercial_df['4_not_otp'] = commercial_df.apply(not_is_otp, axis=1)

    txn_counts_4 = commercial_df['4_not_otp'].value_counts()
    print(f'After OTP exclusion:')
    print(txn_counts_4)
    print(f'Percentage retained: {(txn_counts_4.get(True, 0) / len(commercial_df)) * 100:.2f}%\n')

    likely_transactions = commercial_df[commercial_df['4_not_otp'] == True]
    non_transactions = commercial_df[commercial_df['4_not_otp'] == False]

    output_path = 'RESULTS/4_transactions_split.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        likely_transactions.to_excel(writer, sheet_name='likely_transactions', index=False)
        non_transactions.to_excel(writer, sheet_name='non_transactions', index=False)

    print(f'Saved {len(likely_transactions)} likely transactions and {len(non_transactions)} non-transactions to {output_path}')

    # ==========================================
    # STAGE 5: COLLECT REQUEST EXCLUSION
    # ==========================================
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
        if not isinstance(text, str):
            return False
        if not row['4_not_otp']:
            return False
        return not bool(collect_pattern.search(text))

    print('\nSTAGE 5: Applying collect/mandate request exclusion...')
    commercial_df['5_not_collect'] = commercial_df.apply(not_is_collect, axis=1)

    txn_counts_5 = commercial_df['5_not_collect'].value_counts()
    print(f'After collect request exclusion:')
    print(txn_counts_5)
    print(f'Percentage retained: {(txn_counts_5.get(True, 0) / len(commercial_df)) * 100:.2f}%\n')

    likely_transactions = commercial_df[commercial_df['5_not_collect'] == True]
    non_transactions = commercial_df[commercial_df['5_not_collect'] == False]

    output_path = 'RESULTS/5_transactions_split.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        likely_transactions.to_excel(writer, sheet_name='likely_transactions', index=False)
        non_transactions.to_excel(writer, sheet_name='non_transactions', index=False)

    print(f'Saved {len(likely_transactions)} likely transactions and {len(non_transactions)} non-transactions to {output_path}')
    print(f'\n--- PIPELINE SUMMARY ---')
    print(f'Total commercial/shortcode messages: {len(commercial_df)}')
    print(f'Stage 1 (has amount):        {commercial_df["1_has_amount"].sum()}')
    print(f'Stage 2 (has account/card):   {commercial_df["2_has_acct"].sum()}')
    print(f'Stage 3 (has txn verb):       {commercial_df["3_has_verb"].sum()}')
    print(f'Stage 4 (not OTP):            {commercial_df["4_not_otp"].sum()}')
    print(f'Stage 5 (not collect req):    {commercial_df["5_not_collect"].sum()}')

if __name__ == "__main__":
    main()