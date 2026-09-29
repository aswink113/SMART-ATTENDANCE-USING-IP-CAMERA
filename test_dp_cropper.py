import unittest
import base64
import os
import io
from PIL import Image

from app import app
from database import get_employee_by_id

class TestDpCropperAndStaticMedia(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess['role'] = 'admin'
            sess['user_id'] = 1
            sess['username'] = 'admin'

    def test_01_serve_dataset_photo(self):
        # Request ASWIN K's photo
        res = self.client.get('/Dataset/FTM118_ASWIN_K/photo_1_1789386562.jpg')
        self.assertEqual(res.status_code, 200)
        self.assertIn('image', res.content_type)
        print("[PASS] Successfully served dataset image file from /Dataset/ path (HTTP 200)")

    def test_02_crop_and_save_dp(self):
        # Create a small 100x100 RGB dummy image in memory
        img = Image.new('RGB', (100, 100), color=(99, 102, 241))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        b64_str = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

        res = self.client.post('/api/employee/dp/save', json={
            'emp_id': 'FTM118',
            'image': b64_str
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('dp_', data['dp_url'])
        print(f"[PASS] DP saved successfully: {data['dp_url']}")

        # Verify in DB
        emp = get_employee_by_id('FTM118')
        self.assertIn('dp_', emp['profile_photo'])
        print(f"[PASS] Database verified profile_photo = {emp['profile_photo']}")

        # Verify serving the newly cropped DP
        dp_res = self.client.get('/' + emp['profile_photo'])
        self.assertEqual(dp_res.status_code, 200)
        print("[PASS] Successfully fetched newly cropped DP from Flask static route (HTTP 200)")

if __name__ == '__main__':
    unittest.main()
