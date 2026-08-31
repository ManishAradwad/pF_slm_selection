"""
Unit tests for Indian SMS taxonomy classification and prefilter simulations.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.prefilter_simulator import (
    run_android_prefilter,
    run_ios_prefilter,
    run_unified_prefilter,
)
from lfm25.taxonomy import CATEGORIES_METADATA, classify_sms_record


class TestDatasetSegregation(unittest.TestCase):
    def test_taxonomy_metadata_integrity(self) -> None:
        """Verify all defined categories have valid primary categories and ground truth targets."""
        for sub_id, meta in CATEGORIES_METADATA.items():
            self.assertIn(
                meta["primary"],
                ["TRANSACTIONAL", "EDGE_FINANCIAL", "NON_TRANSACTIONAL", "UNCATEGORIZED"],
            )
            self.assertIn(meta["action_type"], ["debit", "credit", "none", "unknown"])
            self.assertIn(meta["ground_truth_target"], ["transaction_json", "null"])

    def test_card_debit_classification(self) -> None:
        """Test Credit/Debit card purchase patterns."""
        sms = "Alert! You've spent Rs.194 On HDFC Bank Debit Card xx4955 At Swiggy On 2023-09-15:12:06:00 Avl bal: 4804 Not you?Call 18002586161"
        res = classify_sms_record(sender="AD-HDFCBK", body=sms)
        self.assertEqual(res.primary_category, "TRANSACTIONAL")
        self.assertEqual(res.subcategory, "tx.debit.card")
        self.assertEqual(res.action_type, "debit")

    def test_upi_outward_classification(self) -> None:
        """Test UPI outward debit patterns."""
        sms = "Money Transfer:Rs 239.00 from HDFC Bank A/c **9141 on 16-09-23 to EURONETGPAY UPI: 325945314950 Not you? Call 18002586161"
        res = classify_sms_record(sender="AD-HDFCBK", body=sms)
        self.assertEqual(res.primary_category, "TRANSACTIONAL")
        self.assertEqual(res.subcategory, "tx.debit.upi")
        self.assertEqual(res.action_type, "debit")

    def test_upi_inward_classification(self) -> None:
        """Test Inward UPI credit patterns."""
        sms = "Dear SBI User, your A/c ending with XX1234 has been credited by Rs. 500.00 on 12-Jun-23 by UPI/CR/316200000000/rahul@okaxis. Bal: Rs. 15000"
        res = classify_sms_record(sender="SBIUPI", body=sms)
        self.assertEqual(res.primary_category, "TRANSACTIONAL")
        self.assertEqual(res.subcategory, "tx.credit.upi")
        self.assertEqual(res.action_type, "credit")

    def test_failed_transaction_edge_case(self) -> None:
        """Test that failed/declined transactions are categorized under EDGE_FINANCIAL."""
        sms = "Transaction of Rs. 1,500.00 on your HDFC Bank Card XX1234 was declined due to insufficient limit."
        res = classify_sms_record(sender="HDFCBK", body=sms)
        self.assertEqual(res.primary_category, "EDGE_FINANCIAL")
        self.assertEqual(res.subcategory, "edge.txn_failed_declined")
        self.assertEqual(res.ground_truth_target, "null")

    def test_bill_due_edge_case(self) -> None:
        """Test statement generated reminders are categorized under EDGE_FINANCIAL."""
        sms = "Your HDFC Bank Credit Card statement for card ending 0816 is generated. Total amount due: Rs. 12,450.00, Min due: Rs. 650.00. Due date: 24-Jul-2023."
        res = classify_sms_record(sender="HDFCBK", body=sms)
        self.assertEqual(res.primary_category, "EDGE_FINANCIAL")
        self.assertEqual(res.subcategory, "edge.bill_generated_or_due")
        self.assertEqual(res.ground_truth_target, "null")

    def test_otp_classification(self) -> None:
        """Test OTP verification message classification."""
        sms = "482910 is the OTP for your online transaction of Rs. 1500 on HDFC Bank Card ending 1234. Do not share OTP with anyone."
        res = classify_sms_record(sender="HDFCBK", body=sms)
        self.assertEqual(res.primary_category, "NON_TRANSACTIONAL")
        self.assertEqual(res.subcategory, "non_tx.otp_auth_2fa")

    def test_android_and_ios_prefilter_on_valid_transaction(self) -> None:
        """Test that a valid card transaction passes both Android and iOS prefilters."""
        sender = "VM-HDFCBK"
        sms = "Alert! You've spent Rs.250 on HDFC Bank Debit Card xx4955 at Zomato. Avl bal Rs. 5000."

        android_res = run_android_prefilter(sender, sms)
        self.assertTrue(android_res.passed)

        ios_res = run_ios_prefilter(sender, sms)
        self.assertTrue(ios_res.passed)
        self.assertEqual(ios_res.decision, "eligible")

    def test_ios_prefilter_on_failed_transaction(self) -> None:
        """Test iOS prefilter explicitly rejects failed transactions without dot."""
        sender = "VM-HDFCBK"
        sms = "Txn of INR 500 on Card xx1234 failed due to timeout"

        ios_res = run_ios_prefilter(sender, sms)
        self.assertFalse(ios_res.passed)
        self.assertEqual(ios_res.rejection_code, "unsuccessful_transaction")

    def test_unified_prefilter_on_failed_transaction_with_dots(self) -> None:
        """Test unified prefilter handles real-world Indian formats with dots (e.g. Rs. 500.00)."""
        sender = "VM-HDFCBK"
        sms = "Txn of Rs.500.00 on Card xx1234 was declined due to insufficient limit."

        unified_res = run_unified_prefilter(sender, sms)
        self.assertFalse(unified_res.passed)
        self.assertEqual(unified_res.rejection_stage, "reject_unsuccessful_transaction")

    def test_personal_mobile_sender_rejection(self) -> None:
        """Test that personal 10-digit senders are rejected by Android and Unified filters."""
        sender = "+919876543210"
        sms = "Hey paid you Rs. 500 on card xx1234."

        android_res = run_android_prefilter(sender, sms)
        self.assertFalse(android_res.passed)
        self.assertEqual(android_res.rejection_stage, "reject_personal_mobile_sender")

        unified_res = run_unified_prefilter(sender, sms)
        self.assertFalse(unified_res.passed)
        self.assertEqual(unified_res.rejection_stage, "reject_personal_sender")

    def test_unified_contract_spec_file(self) -> None:
        """Test that the unified prefilter contract specification JSON exists and is valid."""
        spec_path = REPO_ROOT / "configs" / "contracts" / "unified-prefilter-spec-v1.json"
        self.assertTrue(spec_path.exists())
        with spec_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["contract_id"], "unified-prefilter-spec-v1")
        self.assertEqual(len(data["ordered_stages"]), 10)


if __name__ == "__main__":
    unittest.main()
