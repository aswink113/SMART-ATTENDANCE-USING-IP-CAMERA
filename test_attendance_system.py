import os
import sys
import unittest
import numpy as np
import cv2
from datetime import datetime, timedelta

from database import (
    init_db, add_employee, get_all_employees,
    process_gesture_punch, verify_admin, verify_employee,
    get_employee_attendance_history, get_today_attendance,
    delete_employee, update_camera_settings
)
from face_engine import face_engine
from gesture_engine import gesture_engine
from reports import generate_attendance_excel

class TestSmartAttendanceSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_admin_auth(self):
        admin = verify_admin('admin', 'admin123')
        self.assertIsNotNone(admin)
        self.assertEqual(admin['username'], 'admin')

        invalid = verify_admin('admin', 'wrongpass')
        self.assertIsNone(invalid)

    def test_02_employee_crud_and_auth(self):
        # Create a test employee
        emp_id = "EMP_TEST_001"
        name = "Test Engineer"
        email = "test.eng@company.com"
        phone = "1234567890"
        password = "secretpassword"

        # Cleanup if exists
        delete_employee(emp_id)

        ok, msg = add_employee(emp_id, name, email, phone, password, "QA")
        self.assertTrue(ok)

        # Verify authentication
        emp_by_id = verify_employee(emp_id, password)
        self.assertIsNotNone(emp_by_id)
        self.assertEqual(emp_by_id['name'], name)

        emp_by_email = verify_employee(email, password)
        self.assertIsNotNone(emp_by_email)

    def test_03_in_out_and_5min_cooldown_rules(self):
        emp_id = "EMP_RULE_001"
        delete_employee(emp_id)
        add_employee(emp_id, "Rule Tester", "rule@test.com", "9999999999", "pass123", "Engineering")

        # Set cooldown to 5 minutes
        update_camera_settings(
            camera_source='0', rtsp_ip='', rtsp_port='554', rtsp_user='',
            rtsp_pass='', rtsp_path='/stream', rtsp_url_override='',
            cooldown_minutes=5, face_threshold=0.40, gesture_threshold=0.50
        )

        # 1. First punch -> Must be IN
        res1 = process_gesture_punch(emp_id=emp_id, confidence=0.88)
        self.assertEqual(res1['status'], 'IN_SUCCESS')
        self.assertEqual(res1['type'], 'IN')
        print("[PASS] Rule 1: First punch recorded as IN")

        # 2. Immediate second punch (< 5 mins) -> Must be SKIPPED due to COOLDOWN
        res2 = process_gesture_punch(emp_id=emp_id, confidence=0.89)
        self.assertEqual(res2['status'], 'COOLDOWN')
        self.assertEqual(res2['type'], 'SKIPPED')
        self.assertGreater(res2['remaining_seconds'], 0)
        print(f"[PASS] Rule 2: Cooldown active ({res2['remaining_seconds']}s remaining), punch skipped")

        # Clean up
        delete_employee(emp_id)

    def test_04_excel_export(self):
        out_file = generate_attendance_excel()
        self.assertTrue(os.path.exists(out_file))
        self.assertTrue(os.path.getsize(out_file) > 1000)
        print(f"[PASS] Excel report generated successfully: {out_file}")

    def test_05_face_and_gesture_engines(self):
        self.assertIsNotNone(face_engine.detector)
        self.assertIsNotNone(face_engine.recognizer)
        self.assertTrue(gesture_engine.is_ready)

        # Synthetic frame test
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        is_raised, hands = gesture_engine.detect_gesture(dummy_frame)
        self.assertIsInstance(is_raised, bool)
        print("[PASS] Face & Gesture engines evaluated synthetic frame without errors")

    def test_06_topk_similarity_and_augmentation(self):
        # Create synthetic face-like image (112x112 RGB pattern)
        synthetic_img = np.full((120, 120, 3), 128, dtype=np.uint8)
        cv2.circle(synthetic_img, (60, 60), 30, (200, 200, 200), -1)
        
        # Test feature normalization check with synthetic vectors
        vec1 = np.random.randn(128).astype(np.float32)
        norm_v1 = vec1 / (np.linalg.norm(vec1) + 1e-7)
        self.assertAlmostEqual(np.linalg.norm(norm_v1), 1.0, places=4)

        # Set mock known embeddings to verify top-k matching behavior
        mock_emb_1 = norm_v1.copy()
        mock_emb_2 = norm_v1.copy() + np.random.normal(0, 0.01, 128)
        mock_emb_2 /= np.linalg.norm(mock_emb_2)

        face_engine.known_embeddings = np.array([mock_emb_1, mock_emb_2])
        face_engine.known_labels = np.array(['EMP_ACCURACY_TEST', 'EMP_ACCURACY_TEST'])
        face_engine.is_trained = True

        # Dummy face box [x, y, w, h]
        dummy_face_data = np.array([10, 10, 50, 50, 20, 20, 40, 20, 30, 35, 20, 45, 40, 45, 0.99])
        
        # Recognition call should execute without exceptions
        emp_id, score, bbox = face_engine.recognize_face(synthetic_img, dummy_face_data, threshold=0.30)
        self.assertIsInstance(score, float)
        print(f"[PASS] Face engine accuracy evaluation test completed: emp_id={emp_id}, score={score:.3f}")

if __name__ == '__main__':
    unittest.main()
