import unittest
from app import app
from database import init_db, add_employee

class TestAppRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()
        # Seed a test employee if needed
        add_employee('TEST001', 'Test Employee', 'testemp@company.com', '9999999999', 'password123', 'Engineering')

    def test_login_page_renders(self):
        res = self.client.get('/login')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'VisionAttend', res.data)
        self.assertIn(b'Sign In', res.data)

    def test_admin_login_and_dashboard(self):
        # Post admin login with unified form
        res = self.client.post('/login', data={
            'username': 'admin',
            'password': 'admin123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Live Camera Attendance Monitoring', res.data)

    def test_employee_login_by_id(self):
        # Post employee login using emp_id
        res = self.client.post('/login', data={
            'username': 'TEST001',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Employee Portal', res.data)

    def test_employee_login_by_email(self):
        # Post employee login using email
        res = self.client.post('/login', data={
            'username': 'testemp@company.com',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Employee Portal', res.data)

    def test_invalid_login(self):
        # Post invalid credentials
        res = self.client.post('/login', data={
            'username': 'wronguser',
            'password': 'wrongpassword'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Invalid credentials', res.data)

    def test_admin_subpages(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'admin'
            sess['username'] = 'admin'
            sess['user_id'] = 1

        res_emp = self.client.get('/admin/employees')
        self.assertEqual(res_emp.status_code, 200)
        self.assertIn(b'Employee Management', res_emp.data)

        res_att = self.client.get('/admin/attendance')
        self.assertEqual(res_att.status_code, 200)
        self.assertIn(b'Attendance Master Logs', res_att.data)

        res_cam = self.client.get('/admin/camera')
        self.assertEqual(res_cam.status_code, 200)
        self.assertIn(b'IP Camera Stream Configuration', res_cam.data)

    def test_api_live_status(self):
        res = self.client.get('/api/live_status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('camera_status', data)

    def test_change_employee_id(self):
        # Register a test employee to change ID
        add_employee('IDCHANGE01', 'Change Name', 'idchange@company.com', '1234567890', 'pass123', 'Testing')
        with self.client.session_transaction() as sess:
            sess['role'] = 'admin'
            sess['username'] = 'admin'
            sess['user_id'] = 1

        # Request ID change
        res = self.client.post('/api/employee/edit', data={
            'emp_id': 'IDCHANGE01',
            'new_emp_id': 'IDCHANGE02',
            'name': 'Change Name Updated',
            'email': 'idchange@company.com',
            'phone': '1234567890',
            'department': 'Testing'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])

        # Verify old ID is gone and new ID works
        res_old = self.client.post('/login', data={'username': 'IDCHANGE01', 'password': 'pass123'}, follow_redirects=True)
        self.assertIn(b'Invalid credentials', res_old.data)

    def test_camera_api_endpoints(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'admin'
            sess['username'] = 'admin'
            sess['user_id'] = 1

        # Test camera save
        res_save = self.client.post('/api/camera/save', data={
            'camera_source': 'rtsp',
            'rtsp_host': '192.168.1.72',
            'rtsp_port': '554',
            'rtsp_user': 'admin',
            'rtsp_pass': 'Admin@123',
            'rtsp_path': '/stream',
            'face_threshold': '0.4',
            'cooldown_minutes': '5',
            'gesture_threshold': '0.5'
        })
        self.assertEqual(res_save.status_code, 200)
        data_save = res_save.get_json()
        self.assertTrue(data_save['success'])

        # Test camera test probe (graceful network failure handling on unreachable IP)
        res_test = self.client.post('/api/camera/test', data={
            'camera_source': 'rtsp',
            'rtsp_ip': '192.0.2.1', # RFC 5737 TEST-NET address
            'rtsp_port': '554',
            'rtsp_user': 'admin',
            'rtsp_pass': 'pass',
            'rtsp_path': '/stream'
        })
        self.assertEqual(res_test.status_code, 200)
        data_test = res_test.get_json()
        self.assertIn('success', data_test)
        self.assertIn('message', data_test)

if __name__ == '__main__':
    unittest.main()
