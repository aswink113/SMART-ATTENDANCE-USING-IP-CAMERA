import os
import io
import base64
import time
from datetime import datetime, date
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, Response, send_file, send_from_directory, flash
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from config import SECRET_KEY, DATASET_DIR, CAPTURES_DIR
from database import (
    init_db, verify_admin, verify_employee,
    add_employee, get_all_employees, get_employee_by_id,
    update_employee, delete_employee, update_employee_photos,
    update_employee_profile, update_employee_dp,
    get_today_attendance, get_employee_attendance_history,
    get_all_attendance_records, get_recent_punch_logs,
    get_admin_dashboard_stats, get_camera_settings,
    update_camera_settings, process_gesture_punch,
    get_machine_analytics_summary
)
from face_engine import face_engine
from camera_stream import camera_stream, CameraStreamManager
from reports import generate_attendance_excel

# Initialize database schema
init_db()

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ----------------- Auth Decorators -----------------

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Please log in as an administrator to access this page.', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def employee_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'employee':
            flash('Please log in with your employee account to view your portal.', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ----------------- Authentication Routes -----------------

@app.route('/')
def index():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif session.get('role') == 'employee':
        return redirect(url_for('employee_dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # 1. Attempt admin verification
        admin = verify_admin(username, password)
        if admin:
            session['role'] = 'admin'
            session['user_id'] = admin['id']
            session['username'] = admin['username']
            return redirect(url_for('admin_dashboard'))

        # 2. Attempt employee verification (by Employee ID or Email)
        emp = verify_employee(username, password)
        if emp:
            session['role'] = 'employee'
            session['user_id'] = emp['id']
            session['emp_id'] = emp['emp_id']
            session['name'] = emp['name']
            session['email'] = emp['email']
            return redirect(url_for('employee_dashboard'))

        flash('Invalid credentials. Please verify your username, Employee ID, or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login_page'))

# ----------------- Static Media Serving Routes -----------------

@app.route('/Dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/dataset/<path:filename>')
def serve_dataset_lower(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/captures/<path:filename>')
def serve_captures(filename):
    return send_from_directory(CAPTURES_DIR, filename)

# ----------------- Video Streaming Route -----------------

@app.route('/video_feed')
def video_feed():
    return Response(
        camera_stream.generate_mjpeg_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ----------------- Admin Routes -----------------

@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = get_admin_dashboard_stats()
    today_records = get_today_attendance()
    recent_punches = get_recent_punch_logs(15)
    settings = get_camera_settings()
    machine_analytics = get_machine_analytics_summary()
    return render_template(
        'admin_dashboard.html',
        stats=stats,
        today_records=today_records,
        recent_punches=recent_punches,
        settings=settings,
        machine_analytics=machine_analytics,
        active_page='dashboard'
    )

@app.route('/admin/employees')
@admin_required
def admin_employees():
    employees = get_all_employees()
    return render_template(
        'admin_employees.html',
        employees=employees,
        active_page='employees',
        is_model_trained=face_engine.is_trained,
        known_samples=len(face_engine.known_embeddings)
    )

@app.route('/admin/attendance')
@admin_required
def admin_attendance():
    start_date = request.args.get('start_date', date.today().strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', date.today().strftime('%Y-%m-%d'))
    department = request.args.get('department', 'All')
    records = get_all_attendance_records(start_date=start_date, end_date=end_date, department=department)
    employees = get_all_employees()
    departments = sorted(list(set(e.get('department', 'General') for e in employees if e.get('department'))))
    
    return render_template(
        'admin_attendance.html',
        records=records,
        start_date=start_date,
        end_date=end_date,
        department=department,
        departments=departments,
        active_page='attendance'
    )

@app.route('/admin/camera')
@admin_required
def admin_camera():
    settings = get_camera_settings()
    return render_template(
        'admin_camera.html',
        settings=settings,
        active_page='camera',
        camera_status=camera_stream.camera_status,
        camera_fps=camera_stream.fps,
        camera_source=camera_stream.camera_source_str
    )

# ----------------- Employee Portal Routes -----------------

@app.route('/employee')
@employee_required
def employee_dashboard():
    emp_id = session.get('emp_id')
    emp_info = get_employee_by_id(emp_id)
    today_str = date.today().strftime('%Y-%m-%d')
    history = get_employee_attendance_history(emp_id)
    
    # Calculate summary metrics
    total_days_worked = len(history)
    total_hours_worked = round(sum(h.get('total_hours', 0.0) or 0.0 for h in history), 1)
    avg_hours = round(total_hours_worked / total_days_worked, 1) if total_days_worked > 0 else 0.0

    today_record = next((h for h in history if h['date'] == today_str), None)

    return render_template(
        'employee_dashboard.html',
        employee=emp_info,
        today_record=today_record,
        history=history,
        total_days=total_days_worked,
        total_hours=total_hours_worked,
        avg_hours=avg_hours,
        active_page='attendance'
    )

@app.route('/employee/profile')
@employee_required
def employee_profile():
    emp_id = session.get('emp_id')
    emp_info = get_employee_by_id(emp_id)
    return render_template(
        'employee_profile.html',
        employee=emp_info,
        active_page='profile'
    )

# ----------------- API Endpoints -----------------

@app.route('/api/live_status')
def api_live_status():
    latest = camera_stream.latest_punch_event
    event_age = time.time() - camera_stream.latest_event_time
    return jsonify({
        'camera_status': camera_stream.camera_status,
        'fps': camera_stream.fps,
        'latest_event': latest if event_age < 4.0 else None,
        'is_model_trained': face_engine.is_trained
    })

@app.route('/api/camera/stream_frame', methods=['POST'])
def api_stream_frame():
    try:
        data = request.get_json() or {}
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        if 'base64,' in image_b64:
            image_b64 = image_b64.split('base64,')[1]
        
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            camera_stream.feed_client_frame(frame)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid frame'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/mode', methods=['POST'])
def api_camera_mode():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'server')
        if mode == 'browser':
            camera_stream.pause_hardware()
            return jsonify({'success': True, 'mode': 'browser', 'message': 'Hardware camera released for browser'})
        else:
            camera_stream.resume_hardware()
            return jsonify({'success': True, 'mode': 'server', 'message': 'Hardware camera capture resumed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/stats')
@admin_required
def api_admin_stats():
    return jsonify(get_admin_dashboard_stats())

@app.route('/api/employee/add', methods=['POST'])
@admin_required
def api_add_employee():
    try:
        emp_id = request.form.get('emp_id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '').strip()
        department = request.form.get('department', 'General').strip()

        if not emp_id or not name or not email or not password:
            return jsonify({'success': False, 'message': 'All required fields must be filled.'}), 400

        # Target folder for employee photos
        emp_folder = os.path.join(DATASET_DIR, f"{emp_id}_{secure_filename(name)}")
        os.makedirs(emp_folder, exist_ok=True)
        photo_paths = []

        # Check for uploaded file photos (up to 3)
        files = request.files.getlist('photos')
        for i, file in enumerate(files):
            if file and file.filename != '':
                filename = f"photo_{i+1}_{int(time.time())}.jpg"
                save_path = os.path.join(emp_folder, filename)
                file.save(save_path)
                photo_paths.append(save_path)

        # Check for base64 webcam captures from modal
        for i in range(1, 4):
            b64_data = request.form.get(f'webcam_photo_{i}')
            if b64_data and 'base64,' in b64_data:
                img_bytes = base64.b64decode(b64_data.split('base64,')[1])
                filename = f"capture_{i}_{int(time.time())}.jpg"
                save_path = os.path.join(emp_folder, filename)
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)
                photo_paths.append(save_path)

        if len(photo_paths) == 0:
            return jsonify({'success': False, 'message': 'Please provide at least 1 (recommended 3) photos of the employee.'}), 400

        # Save to database
        ok, msg = add_employee(emp_id, name, email, phone, password, department, photo_paths)
        if not ok:
            return jsonify({'success': False, 'message': msg}), 400

        # Automatically train/update face model
        train_ok, train_msg = face_engine.train_model()

        return jsonify({
            'success': True,
            'message': f'Employee registered successfully! {train_msg}'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/employee/edit', methods=['POST'])
@admin_required
def api_edit_employee():
    try:
        data = request.get_json(silent=True) or {}
        emp_id = (data.get('emp_id') or request.form.get('emp_id', '')).strip()
        new_emp_id = (data.get('new_emp_id') or request.form.get('new_emp_id', '')).strip() or emp_id
        name = (data.get('name') or request.form.get('name', '')).strip()
        email = (data.get('email') or request.form.get('email', '')).strip()
        phone = (data.get('phone') or request.form.get('phone', '')).strip()
        department = (data.get('department') or request.form.get('department', 'General')).strip()
        new_password = (data.get('new_password') or request.form.get('new_password', '')).strip()

        if not emp_id or not name or not new_emp_id:
            return jsonify({'success': False, 'message': 'Employee ID and Name are required.'}), 400

        ok, msg = update_employee(emp_id, name, email, phone, department, new_password if new_password else None, new_emp_id=new_emp_id)
        if ok and (new_emp_id != emp_id or name):
            try:
                face_engine.train_model()
            except Exception as train_err:
                print(f"Warning on retrain after edit: {train_err}")

        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/employee/delete', methods=['POST'])
@admin_required
def api_delete_employee():
    try:
        data = request.get_json(silent=True) or {}
        emp_id = data.get('emp_id') or request.form.get('emp_id')
        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID required.'}), 400

        ok, msg = delete_employee(emp_id)
        if ok:
            try:
                face_engine.train_model()
            except Exception as train_err:
                print(f"Warning: Retrain after delete encountered: {train_err}")
        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/employee/profile/update', methods=['POST'])
@employee_required
def api_update_employee_profile():
    try:
        emp_id = session.get('emp_id')
        phone = request.form.get('phone', '').strip()
        new_password = request.form.get('new_password', '').strip()

        if not phone:
            return jsonify({'success': False, 'message': 'Phone number cannot be empty.'}), 400

        ok, msg = update_employee_profile(emp_id, phone, new_password if new_password else None)
        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/employee/photos/upload', methods=['POST'])
def api_upload_employee_photos():
    try:
        # Check authentication: Admin can specify any emp_id, Employee can only update own
        emp_id = request.form.get('emp_id', '').strip()
        if session.get('role') == 'employee':
            emp_id = session.get('emp_id')
        elif session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized access.'}), 403

        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID is required.'}), 400

        emp = get_employee_by_id(emp_id)
        if not emp:
            return jsonify({'success': False, 'message': f'Employee {emp_id} not found.'}), 404

        emp_name = emp['name']
        emp_folder = os.path.join(DATASET_DIR, f"{emp_id}_{secure_filename(emp_name)}")
        os.makedirs(emp_folder, exist_ok=True)
        photo_paths = []

        # Check for uploaded file photos (up to 3)
        files = request.files.getlist('photos')
        for i, file in enumerate(files):
            if file and file.filename != '':
                filename = f"photo_{i+1}_{int(time.time())}.jpg"
                save_path = os.path.join(emp_folder, filename)
                file.save(save_path)
                photo_paths.append(save_path)

        # Check for base64 webcam captures from modal
        for i in range(1, 4):
            b64_data = request.form.get(f'webcam_photo_{i}')
            if b64_data and 'base64,' in b64_data:
                img_bytes = base64.b64decode(b64_data.split('base64,')[1])
                filename = f"capture_{i}_{int(time.time())}.jpg"
                save_path = os.path.join(emp_folder, filename)
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)
                photo_paths.append(save_path)

        if len(photo_paths) == 0:
            return jsonify({'success': False, 'message': 'Please provide at least 1 photo.'}), 400

        # Replace or update employee photos
        ok, msg = update_employee_photos(emp_id, photo_paths, replace_all=True)
        if not ok:
            return jsonify({'success': False, 'message': msg}), 500

        # Retrain face recognition model
        train_ok, train_msg = face_engine.train_model()

        return jsonify({
            'success': True,
            'message': f'Profile & face photos updated successfully! {train_msg}',
            'photos': [p.replace('\\', '/') for p in photo_paths]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/employee/dp/save', methods=['POST'])
def api_save_employee_dp():
    try:
        data = request.get_json() or {}
        emp_id = data.get('emp_id', '').strip()
        image_b64 = data.get('image', '').strip()

        if session.get('role') == 'employee':
            emp_id = session.get('emp_id')
        elif session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized access.'}), 403

        if not emp_id or not image_b64:
            return jsonify({'success': False, 'message': 'Employee ID and image data are required.'}), 400

        emp = get_employee_by_id(emp_id)
        if not emp:
            return jsonify({'success': False, 'message': f'Employee {emp_id} not found.'}), 404

        if 'base64,' in image_b64:
            image_b64 = image_b64.split('base64,')[1]

        img_bytes = base64.b64decode(image_b64)
        emp_folder = os.path.join(DATASET_DIR, f"{emp_id}_{secure_filename(emp['name'])}")
        os.makedirs(emp_folder, exist_ok=True)

        dp_filename = f"dp_{int(time.time())}.jpg"
        dp_full_path = os.path.join(emp_folder, dp_filename)
        with open(dp_full_path, 'wb') as f:
            f.write(img_bytes)

        rel_path = f"Dataset/{emp_id}_{secure_filename(emp['name'])}/{dp_filename}"
        ok, msg = update_employee_dp(emp_id, rel_path)
        
        return jsonify({
            'success': ok,
            'message': 'Profile picture cropped and saved successfully!',
            'dp_url': rel_path
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/employee/retrain', methods=['POST'])
@admin_required
def api_retrain_model():
    ok, msg = face_engine.train_model()
    return jsonify({'success': ok, 'message': msg})

@app.route('/api/camera/save', methods=['POST'])
@admin_required
def api_save_camera():
    try:
        camera_source = request.form.get('camera_source', '0')
        rtsp_ip = request.form.get('rtsp_ip', '')
        rtsp_port = request.form.get('rtsp_port', '554')
        rtsp_user = request.form.get('rtsp_user', '')
        rtsp_pass = request.form.get('rtsp_pass', '')
        rtsp_path = request.form.get('rtsp_path', '/h264Preview_01_main')
        rtsp_url_override = request.form.get('rtsp_url_override', '')
        cooldown_minutes = request.form.get('cooldown_minutes', 5)
        face_threshold = request.form.get('face_threshold', 0.40)
        gesture_threshold = request.form.get('gesture_threshold', 0.50)
        
        enable_machine_detection = 1 if (request.form.get('enable_machine_detection') in ['1', 'true', 'on']) else 0
        enable_dwell_tracking = 1 if (request.form.get('enable_dwell_tracking') in ['1', 'true', 'on']) else 0
        enable_apparel_color = 1 if (request.form.get('enable_apparel_color') in ['1', 'true', 'on']) else 0

        update_camera_settings(
            camera_source, rtsp_ip, rtsp_port, rtsp_user, rtsp_pass,
            rtsp_path, rtsp_url_override, cooldown_minutes, face_threshold, gesture_threshold,
            enable_machine_detection=enable_machine_detection,
            enable_dwell_tracking=enable_dwell_tracking,
            enable_apparel_color=enable_apparel_color
        )

        # Restart camera stream with new source
        camera_stream.restart()

        return jsonify({'success': True, 'message': 'Settings & AI Addon modules saved and reloaded!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/test', methods=['POST'])
@admin_required
def api_test_camera():
    try:
        data = request.get_json(silent=True) or request.form or {}
        ip = data.get('rtsp_ip') or data.get('rtsp_host') or ''
        port = data.get('rtsp_port', '554')
        user = data.get('rtsp_user', '')
        password = data.get('rtsp_pass', '')
        path = data.get('rtsp_path', '')
        url_override = data.get('rtsp_url_override', '')

        result = CameraStreamManager.test_ip_camera(ip, port, user, password, path, url_override)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': f"Test error: {str(e)}"}), 500

@app.route('/api/manual_punch', methods=['POST'])
@admin_required
def api_manual_punch():
    try:
        data = request.get_json() or {}
        emp_id = data.get('emp_id')
        if not emp_id:
            return jsonify({'success': False, 'message': 'Employee ID is required.'}), 400

        result = process_gesture_punch(emp_id=emp_id, snapshot_path=None, confidence=1.0)
        camera_stream.latest_punch_event = result
        camera_stream.latest_event_time = time.time()

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/attendance/export')
@admin_required
def api_export_excel():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    department = request.args.get('department')
    filepath = generate_attendance_excel(start_date=start_date, end_date=end_date, department=department)
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("==================================================================")
    print("  [*] Smart IP Camera Attendance Dashboard is launching...")
    print("  [*] Access the web portal at: http://127.0.0.1:5000")
    print("  [*] Admin Login: admin / admin123")
    print("==================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
