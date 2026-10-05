import sqlite3
import os
from datetime import datetime, date, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from config import DB_PATH, DEFAULT_COOLDOWN_MINUTES, DEFAULT_FACE_THRESHOLD, DEFAULT_GESTURE_THRESHOLD

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
    except Exception:
        pass
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Admins Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Employees Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            department TEXT DEFAULT 'General',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Employee Photos Table (Stores 3 photo paths per employee)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employee_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT NOT NULL,
            photo_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
        )
    ''')

    # Attendance Records Table (Day-by-day record per employee)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT NOT NULL,
            date TEXT NOT NULL,
            in_time TEXT,
            out_time TEXT,
            total_hours REAL DEFAULT 0.0,
            status TEXT DEFAULT 'IN_PROGRESS',
            in_snapshot TEXT,
            out_snapshot TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(emp_id, date),
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
        )
    ''')

    # Punch Logs (Raw audit trail for every punch attempt / gesture event)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS punch_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            punch_type TEXT NOT NULL, -- 'IN', 'OUT', 'SKIPPED_COOLDOWN'
            confidence REAL,
            snapshot_path TEXT,
            status_note TEXT,
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
        )
    ''')

    # Camera & System Settings Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            camera_source TEXT DEFAULT '0', -- '0', '1' or 'rtsp'
            rtsp_ip TEXT DEFAULT '',
            rtsp_port TEXT DEFAULT '554',
            rtsp_user TEXT DEFAULT '',
            rtsp_pass TEXT DEFAULT '',
            rtsp_path TEXT DEFAULT '/h264Preview_01_main',
            rtsp_url_override TEXT DEFAULT '',
            cooldown_minutes INTEGER DEFAULT 5,
            face_threshold REAL DEFAULT 0.40,
            gesture_threshold REAL DEFAULT 0.50,
            enable_machine_detection INTEGER DEFAULT 1,
            enable_dwell_tracking INTEGER DEFAULT 1,
            enable_apparel_color INTEGER DEFAULT 1
        )
    ''')

    # Machine Front Pass & Analytics Events Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS machine_pass_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT DEFAULT 'Unknown',
            person_track_id INTEGER DEFAULT 0,
            date TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            hour_of_day INTEGER NOT NULL,
            dwell_seconds REAL DEFAULT 0.0,
            top_dress_color TEXT DEFAULT 'Unknown',
            status_category TEXT DEFAULT 'PASSED' -- 'PASSED' or 'LOITERING/HIGH_DWELL'
        )
    ''')

    # Ensure profile_photo column exists in employees table
    try:
        cursor.execute('ALTER TABLE employees ADD COLUMN profile_photo TEXT DEFAULT ""')
    except Exception:
        pass

    # Seed Default Admin if not exists
    cursor.execute('SELECT * FROM admins WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        cursor.execute(
            'INSERT INTO admins (username, password_hash) VALUES (?, ?)',
            ('admin', generate_password_hash('admin123'))
        )

    # Seed Default Camera Settings if not exists
    cursor.execute('SELECT * FROM camera_settings WHERE id = 1')
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO camera_settings (
                id, camera_source, rtsp_ip, rtsp_port, rtsp_user, rtsp_pass, rtsp_path, 
                rtsp_url_override, cooldown_minutes, face_threshold, gesture_threshold
            ) VALUES (1, '0', '', '554', '', '', '/h264Preview_01_main', '', ?, ?, ?)
        ''', (DEFAULT_COOLDOWN_MINUTES, DEFAULT_FACE_THRESHOLD, DEFAULT_GESTURE_THRESHOLD))

    conn.commit()
    conn.close()

# ----------------- Path Normalization Helper -----------------

def normalize_photo_path(p):
    if not p:
        return ''
    p = str(p).replace('\\', '/')
    if 'Dataset/' in p:
        return 'Dataset/' + p.split('Dataset/', 1)[1]
    if 'captures/' in p:
        return 'captures/' + p.split('captures/', 1)[1]
    return p

# ----------------- Auth Functions -----------------

def verify_admin(username, password):
    conn = get_db_connection()
    admin = conn.execute('SELECT * FROM admins WHERE username = ?', (username,)).fetchone()
    conn.close()
    if admin and check_password_hash(admin['password_hash'], password):
        return dict(admin)
    return None

def verify_employee(emp_id_or_email, password):
    conn = get_db_connection()
    emp = conn.execute(
        'SELECT * FROM employees WHERE emp_id = ? OR email = ?',
        (emp_id_or_email, emp_id_or_email)
    ).fetchone()
    conn.close()
    if emp and check_password_hash(emp['password_hash'], password):
        return dict(emp)
    return None

# ----------------- Employee CRUD -----------------

def add_employee(emp_id, name, email, phone, password, department='General', photo_paths=None):
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(password)
        emp_id_clean = emp_id.strip()
        email_clean = email.strip().lower()
        
        # Check if already exists by emp_id or email
        existing = conn.execute('SELECT * FROM employees WHERE emp_id = ? OR email = ?', (emp_id_clean, email_clean)).fetchone()

        clean_photos = [normalize_photo_path(p) for p in photo_paths] if photo_paths else []
        dp_to_set = clean_photos[0] if clean_photos else ''

        if existing:
            # Update existing employee details
            conn.execute('''
                UPDATE employees 
                SET emp_id = ?, name = ?, email = ?, phone = ?, password_hash = ?, department = ?,
                    profile_photo = CASE WHEN (profile_photo IS NULL OR profile_photo = '') THEN ? ELSE profile_photo END
                WHERE id = ?
            ''', (emp_id_clean, name.strip(), email_clean, phone.strip(), password_hash, department.strip(), dp_to_set, existing['id']))
            target_emp_id = emp_id_clean
        else:
            # Insert new employee
            conn.execute('''
                INSERT INTO employees (emp_id, name, email, phone, password_hash, department, profile_photo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (emp_id_clean, name.strip(), email_clean, phone.strip(), password_hash, department.strip(), dp_to_set))
            target_emp_id = emp_id_clean

        if clean_photos and len(clean_photos) > 0:
            conn.execute('DELETE FROM employee_photos WHERE emp_id = ?', (target_emp_id,))
            for p in clean_photos:
                conn.execute('INSERT INTO employee_photos (emp_id, photo_path) VALUES (?, ?)', (target_emp_id, p))

        conn.commit()
        return True, "Employee registered / updated successfully!"
    except Exception as e:
        return False, f"Error saving employee: {str(e)}"
    finally:
        conn.close()

def get_all_employees():
    conn = get_db_connection()
    employees = conn.execute('SELECT * FROM employees ORDER BY id DESC').fetchall()
    result = []
    for emp in employees:
        emp_dict = dict(emp)
        photos = conn.execute('SELECT photo_path FROM employee_photos WHERE emp_id = ?', (emp['emp_id'],)).fetchall()
        normalized_photos = [normalize_photo_path(p['photo_path']) for p in photos]
        emp_dict['photos'] = normalized_photos

        dp = normalize_photo_path(emp_dict.get('profile_photo', ''))
        if not dp and len(normalized_photos) > 0:
            dp = normalized_photos[0]
        emp_dict['profile_photo'] = dp

        result.append(emp_dict)
    conn.close()
    return result

def get_employee_by_id(emp_id):
    conn = get_db_connection()
    emp = conn.execute('SELECT * FROM employees WHERE emp_id = ?', (emp_id,)).fetchone()
    if not emp:
        conn.close()
        return None
    emp_dict = dict(emp)
    photos = conn.execute('SELECT photo_path FROM employee_photos WHERE emp_id = ?', (emp_id,)).fetchall()
    normalized_photos = [normalize_photo_path(p['photo_path']) for p in photos]
    emp_dict['photos'] = normalized_photos

    dp = normalize_photo_path(emp_dict.get('profile_photo', ''))
    if not dp and len(normalized_photos) > 0:
        dp = normalized_photos[0]
    emp_dict['profile_photo'] = dp

    conn.close()
    return emp_dict

def update_employee(emp_id, name, email, phone, department, new_password=None, new_emp_id=None):
    conn = get_db_connection()
    try:
        old_id = emp_id.strip()
        target_id = new_emp_id.strip() if (new_emp_id and new_emp_id.strip()) else old_id
        name_clean = name.strip()
        email_clean = email.strip().lower()
        phone_clean = phone.strip()
        dept_clean = department.strip()

        # Check if new_emp_id already belongs to someone else
        if target_id != old_id:
            existing_id = conn.execute('SELECT * FROM employees WHERE emp_id = ? AND emp_id != ?', (target_id, old_id)).fetchone()
            if existing_id:
                return False, f"Employee ID '{target_id}' is already taken by another employee."

        # Check if email belongs to someone else
        existing_email = conn.execute('SELECT * FROM employees WHERE email = ? AND emp_id != ?', (email_clean, old_id)).fetchone()
        if existing_email:
            return False, f"Email address '{email_clean}' is already registered to another employee."

        # Handle folder rename on disk if target_id != old_id
        if target_id != old_id:
            try:
                from config import DATASET_DIR
                from werkzeug.utils import secure_filename
                if os.path.exists(DATASET_DIR):
                    for folder in os.listdir(DATASET_DIR):
                        if folder.startswith(f"{old_id}_") or folder == old_id:
                            old_dir = os.path.join(DATASET_DIR, folder)
                            new_folder_name = f"{target_id}_{secure_filename(name_clean)}"
                            new_dir = os.path.join(DATASET_DIR, new_folder_name)
                            if os.path.isdir(old_dir):
                                if not os.path.exists(new_dir):
                                    os.rename(old_dir, new_dir)
                                # Update photo paths in database
                                photos = conn.execute('SELECT id, photo_path FROM employee_photos WHERE emp_id = ?', (old_id,)).fetchall()
                                for p in photos:
                                    old_p = p['photo_path']
                                    new_p = old_p.replace(folder, new_folder_name).replace('\\', '/')
                                    conn.execute('UPDATE employee_photos SET photo_path = ? WHERE id = ?', (new_p, p['id']))
                                
                                emp_row = conn.execute('SELECT profile_photo FROM employees WHERE emp_id = ?', (old_id,)).fetchone()
                                if emp_row and emp_row['profile_photo']:
                                    new_dp = emp_row['profile_photo'].replace(folder, new_folder_name).replace('\\', '/')
                                    conn.execute('UPDATE employees SET profile_photo = ? WHERE emp_id = ?', (new_dp, old_id))
                            break
            except Exception as disk_err:
                print(f"Warning on dataset folder rename: {disk_err}")

            # Update child tables with new_emp_id
            conn.execute('UPDATE employee_photos SET emp_id = ? WHERE emp_id = ?', (target_id, old_id))
            conn.execute('UPDATE attendance SET emp_id = ? WHERE emp_id = ?', (target_id, old_id))
            conn.execute('UPDATE punch_logs SET emp_id = ? WHERE emp_id = ?', (target_id, old_id))

        if new_password and new_password.strip():
            conn.execute('''
                UPDATE employees SET emp_id = ?, name = ?, email = ?, phone = ?, department = ?, password_hash = ?
                WHERE emp_id = ?
            ''', (target_id, name_clean, email_clean, phone_clean, dept_clean, generate_password_hash(new_password.strip()), old_id))
        else:
            conn.execute('''
                UPDATE employees SET emp_id = ?, name = ?, email = ?, phone = ?, department = ?
                WHERE emp_id = ?
            ''', (target_id, name_clean, email_clean, phone_clean, dept_clean, old_id))

        conn.commit()
        return True, "Employee updated successfully!"
    except Exception as e:
        conn.rollback()
        return False, f"Error updating employee: {str(e)}"
    finally:
        conn.close()

def update_employee_dp(emp_id, dp_path):
    conn = get_db_connection()
    try:
        clean_path = normalize_photo_path(dp_path)
        conn.execute('UPDATE employees SET profile_photo = ? WHERE emp_id = ?', (clean_path, emp_id))
        conn.commit()
        return True, "Profile picture updated successfully!"
    except Exception as e:
        return False, f"Error updating profile picture: {str(e)}"
    finally:
        conn.close()

def update_employee_photos(emp_id, photo_paths, replace_all=True):
    conn = get_db_connection()
    try:
        clean_photos = [normalize_photo_path(p) for p in photo_paths]
        if replace_all:
            conn.execute('DELETE FROM employee_photos WHERE emp_id = ?', (emp_id,))
        for p in clean_photos:
            conn.execute('INSERT INTO employee_photos (emp_id, photo_path) VALUES (?, ?)', (emp_id, p))
        
        # If no profile_photo set, set first photo as DP
        emp = conn.execute('SELECT profile_photo FROM employees WHERE emp_id = ?', (emp_id,)).fetchone()
        if (not emp or not emp['profile_photo']) and len(clean_photos) > 0:
            conn.execute('UPDATE employees SET profile_photo = ? WHERE emp_id = ?', (clean_photos[0], emp_id))

        conn.commit()
        return True, "Photos updated successfully!"
    except Exception as e:
        return False, f"Error updating photos: {str(e)}"
    finally:
        conn.close()

def update_employee_profile(emp_id, phone, new_password=None):
    conn = get_db_connection()
    try:
        if new_password and new_password.strip():
            conn.execute('''
                UPDATE employees SET phone = ?, password_hash = ?
                WHERE emp_id = ?
            ''', (phone.strip(), generate_password_hash(new_password.strip()), emp_id))
        else:
            conn.execute('''
                UPDATE employees SET phone = ?
                WHERE emp_id = ?
            ''', (phone.strip(), emp_id))
        conn.commit()
        return True, "Profile updated successfully!"
    except Exception as e:
        return False, f"Error updating profile: {str(e)}"
    finally:
        conn.close()

def delete_employee(emp_id):
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM employees WHERE emp_id = ?', (emp_id,))
        conn.execute('DELETE FROM employee_photos WHERE emp_id = ?', (emp_id,))
        conn.execute('DELETE FROM attendance WHERE emp_id = ?', (emp_id,))
        conn.execute('DELETE FROM punch_logs WHERE emp_id = ?', (emp_id,))
        conn.commit()
        return True, "Employee deleted successfully!"
    except Exception as e:
        return False, f"Error deleting employee: {str(e)}"
    finally:
        conn.close()

# ----------------- Core 5-Minute In/Out Attendance Rule Engine -----------------

def process_gesture_punch(emp_id, snapshot_path=None, confidence=0.0):
    """
    Implements the specified business rule:
    1. First punch of the day -> Marks 'IN'
    2. Next 5 minutes -> Skipped (cooldown period)
    3. After 5 minutes -> Marks 'OUT' (or updates the 'OUT' punch if punched later in the day)
    """
    conn = get_db_connection()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Get cooldown setting
    settings = conn.execute('SELECT cooldown_minutes FROM camera_settings WHERE id = 1').fetchone()
    cooldown_min = settings['cooldown_minutes'] if settings else DEFAULT_COOLDOWN_MINUTES
    cooldown_delta = timedelta(minutes=cooldown_min)

    # Check if employee exists
    emp = conn.execute('SELECT name FROM employees WHERE emp_id = ?', (emp_id,)).fetchone()
    if not emp:
        conn.close()
        return {
            'status': 'ERROR',
            'message': f'Employee ID {emp_id} not found in database.',
            'emp_id': emp_id,
            'name': 'Unknown'
        }

    emp_name = emp['name']

    # Check today's attendance record
    att = conn.execute('SELECT * FROM attendance WHERE emp_id = ? AND date = ?', (emp_id, today_str)).fetchone()

    if not att:
        # First punch of the day: RECORD IN
        conn.execute('''
            INSERT INTO attendance (emp_id, date, in_time, status, in_snapshot)
            VALUES (?, ?, ?, 'IN_PROGRESS', ?)
        ''', (emp_id, today_str, time_str, snapshot_path))

        # Log to punch_logs
        conn.execute('''
            INSERT INTO punch_logs (emp_id, timestamp, punch_type, confidence, snapshot_path, status_note)
            VALUES (?, ?, 'IN', ?, ?, 'First punch of the day (IN)')
        ''', (emp_id, f"{today_str} {time_str}", confidence, snapshot_path))

        conn.commit()
        conn.close()

        return {
            'status': 'IN_SUCCESS',
            'type': 'IN',
            'message': f'Welcome {emp_name}! Punch IN recorded at {time_str}',
            'emp_id': emp_id,
            'name': emp_name,
            'time': time_str,
            'cooldown_minutes': cooldown_min
        }

    # Record exists. Determine the last punch time
    in_time_dt = datetime.strptime(f"{today_str} {att['in_time']}", '%Y-%m-%d %H:%M:%S')
    last_punch_dt = in_time_dt

    if att['out_time']:
        out_time_dt = datetime.strptime(f"{today_str} {att['out_time']}", '%Y-%m-%d %H:%M:%S')
        if out_time_dt > last_punch_dt:
            last_punch_dt = out_time_dt

    time_since_last_punch = now - last_punch_dt

    if time_since_last_punch < cooldown_delta:
        # Cooldown active! Skip punch
        remaining_seconds = int((cooldown_delta - time_since_last_punch).total_seconds())
        remaining_mins = remaining_seconds // 60
        remaining_secs = remaining_seconds % 60

        # Log skipped punch
        conn.execute('''
            INSERT INTO punch_logs (emp_id, timestamp, punch_type, confidence, snapshot_path, status_note)
            VALUES (?, ?, 'SKIPPED_COOLDOWN', ?, ?, ?)
        ''', (emp_id, f"{today_str} {time_str}", confidence, snapshot_path,
              f"Skipped due to cooldown. {remaining_mins}m {remaining_secs}s remaining."))

        conn.commit()
        conn.close()

        return {
            'status': 'COOLDOWN',
            'type': 'SKIPPED',
            'message': f'Cooldown Active for {emp_name}. Please wait {remaining_mins}m {remaining_secs}s before punching.',
            'emp_id': emp_id,
            'name': emp_name,
            'time': time_str,
            'remaining_seconds': remaining_seconds
        }

    # More than 5 minutes elapsed since previous punch: RECORD / UPDATE OUT
    total_seconds = (now - in_time_dt).total_seconds()
    total_hours = round(total_seconds / 3600.0, 2)

    conn.execute('''
        UPDATE attendance 
        SET out_time = ?, total_hours = ?, status = 'COMPLETED', out_snapshot = ?, updated_at = CURRENT_TIMESTAMP
        WHERE emp_id = ? AND date = ?
    ''', (time_str, total_hours, snapshot_path or att['out_snapshot'], emp_id, today_str))

    # Log to punch_logs
    conn.execute('''
        INSERT INTO punch_logs (emp_id, timestamp, punch_type, confidence, snapshot_path, status_note)
        VALUES (?, ?, 'OUT', ?, ?, ?)
    ''', (emp_id, f"{today_str} {time_str}", confidence, snapshot_path, f"Punch OUT recorded (Total: {total_hours} hrs)"))

    conn.commit()
    conn.close()

    return {
        'status': 'OUT_SUCCESS',
        'type': 'OUT',
        'message': f'Goodbye {emp_name}! Punch OUT recorded at {time_str} (Worked: {total_hours} hrs)',
        'emp_id': emp_id,
        'name': emp_name,
        'time': time_str,
        'total_hours': total_hours
    }

# ----------------- Attendance Query & Reports -----------------

def get_today_attendance():
    conn = get_db_connection()
    today_str = date.today().strftime('%Y-%m-%d')
    query = '''
        SELECT a.*, e.name, e.department, e.email, e.phone
        FROM attendance a
        JOIN employees e ON a.emp_id = e.emp_id
        WHERE a.date = ?
        ORDER BY a.in_time DESC
    '''
    rows = conn.execute(query, (today_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_employee_attendance_history(emp_id, start_date=None, end_date=None):
    conn = get_db_connection()
    query = '''
        SELECT * FROM attendance 
        WHERE emp_id = ?
    '''
    params = [emp_id]

    if start_date:
        query += ' AND date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND date <= ?'
        params.append(end_date)

    query += ' ORDER BY date DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_attendance_records(start_date=None, end_date=None, department=None, emp_id=None):
    conn = get_db_connection()
    query = '''
        SELECT a.*, e.name, e.department, e.email
        FROM attendance a
        JOIN employees e ON a.emp_id = e.emp_id
        WHERE 1=1
    '''
    params = []
    if start_date:
        query += ' AND a.date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND a.date <= ?'
        params.append(end_date)
    if department and department != 'All':
        query += ' AND e.department = ?'
        params.append(department)
    if emp_id:
        query += ' AND a.emp_id = ?'
        params.append(emp_id)

    query += ' ORDER BY a.date DESC, a.in_time DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_punch_logs(limit=20):
    conn = get_db_connection()
    query = '''
        SELECT p.*, e.name, e.department
        FROM punch_logs p
        JOIN employees e ON p.emp_id = e.emp_id
        ORDER BY p.id DESC
        LIMIT ?
    '''
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_admin_dashboard_stats():
    conn = get_db_connection()
    today_str = date.today().strftime('%Y-%m-%d')

    total_employees = conn.execute('SELECT COUNT(*) as count FROM employees').fetchone()['count']
    present_today = conn.execute('SELECT COUNT(*) as count FROM attendance WHERE date = ?', (today_str,)).fetchone()['count']
    in_progress = conn.execute("SELECT COUNT(*) as count FROM attendance WHERE date = ? AND status = 'IN_PROGRESS'", (today_str,)).fetchone()['count']
    completed_out = conn.execute("SELECT COUNT(*) as count FROM attendance WHERE date = ? AND status = 'COMPLETED'", (today_str,)).fetchone()['count']

    absent_count = max(0, total_employees - present_today)
    attendance_rate = round((present_today / total_employees * 100), 1) if total_employees > 0 else 0

    # Department breakdown
    dept_rows = conn.execute('''
        SELECT e.department, COUNT(DISTINCT a.emp_id) as present_count
        FROM employees e
        LEFT JOIN attendance a ON e.emp_id = a.emp_id AND a.date = ?
        GROUP BY e.department
    ''', (today_str,)).fetchall()
    dept_stats = [dict(r) for r in dept_rows]

    conn.close()
    return {
        'total_employees': total_employees,
        'present_today': present_today,
        'currently_in': in_progress,
        'punched_out': completed_out,
        'absent_count': absent_count,
        'attendance_rate': attendance_rate,
        'department_stats': dept_stats
    }

def get_day_by_day_analytics(start_date=None, end_date=None, department=None, today_only=False):
    conn = get_db_connection()
    today_str = date.today().strftime('%Y-%m-%d')
    
    if today_only:
        start_date = today_str
        end_date = today_str
    
    # Filter conditions for SQL
    where_clauses = []
    params = []
    
    if start_date:
        where_clauses.append("a.date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("a.date <= ?")
        params.append(end_date)
    if department and department != 'All':
        where_clauses.append("e.department = ?")
        params.append(department)
        
    where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    
    # Aggregated Day-by-Day counts
    query = f'''
        SELECT 
            a.date,
            COUNT(DISTINCT a.emp_id) as total_present,
            SUM(CASE WHEN a.status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_count,
            SUM(CASE WHEN a.status = 'IN_PROGRESS' THEN 1 ELSE 0 END) as active_count,
            ROUND(SUM(IFNULL(a.total_hours, 0)), 1) as sum_hours,
            ROUND(AVG(IFNULL(a.total_hours, 0)), 1) as avg_hours
        FROM attendance a
        LEFT JOIN employees e ON a.emp_id = e.emp_id
        {where_str}
        GROUP BY a.date
        ORDER BY a.date ASC
    '''
    rows = conn.execute(query, params).fetchall()
    
    # Overall summary counts for the selected filter context
    summary_query = f'''
        SELECT 
            COUNT(DISTINCT a.emp_id) as unique_employees,
            COUNT(a.id) as total_records,
            SUM(CASE WHEN a.status = 'COMPLETED' THEN 1 ELSE 0 END) as total_completed,
            SUM(CASE WHEN a.status = 'IN_PROGRESS' THEN 1 ELSE 0 END) as total_active,
            ROUND(SUM(IFNULL(a.total_hours, 0)), 1) as total_hours
        FROM attendance a
        LEFT JOIN employees e ON a.emp_id = e.emp_id
        {where_str}
    '''
    summary = conn.execute(summary_query, params).fetchone()
    conn.close()
    
    dates = [r['date'] for r in rows]
    total_present = [r['total_present'] for r in rows]
    completed_counts = [r['completed_count'] for r in rows]
    active_counts = [r['active_count'] for r in rows]
    sum_hours = [r['sum_hours'] for r in rows]
    avg_hours = [r['avg_hours'] for r in rows]
    
    return {
        'dates': dates,
        'total_present': total_present,
        'completed_counts': completed_counts,
        'active_counts': active_counts,
        'sum_hours': sum_hours,
        'avg_hours': avg_hours,
        'summary': dict(summary) if summary else {
            'unique_employees': 0,
            'total_records': 0,
            'total_completed': 0,
            'total_active': 0,
            'total_hours': 0.0
        }
    }


# ----------------- Camera Settings -----------------

def get_camera_settings():
    conn = get_db_connection()
    # Check if addon columns exist (migration check)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(camera_settings)")
    cols = [col['name'] for col in cursor.fetchall()]
    if 'enable_machine_detection' not in cols:
        try:
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_machine_detection INTEGER DEFAULT 1")
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_dwell_tracking INTEGER DEFAULT 1")
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_apparel_color INTEGER DEFAULT 1")
            conn.commit()
        except Exception:
            pass

    settings = conn.execute('SELECT * FROM camera_settings WHERE id = 1').fetchone()
    conn.close()
    return dict(settings) if settings else {}

def update_camera_settings(camera_source, rtsp_ip, rtsp_port, rtsp_user, rtsp_pass, rtsp_path, rtsp_url_override, cooldown_minutes, face_threshold, gesture_threshold, enable_machine_detection=1, enable_dwell_tracking=1, enable_apparel_color=1):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(camera_settings)")
    cols = [col['name'] for col in cursor.fetchall()]
    if 'enable_machine_detection' not in cols:
        try:
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_machine_detection INTEGER DEFAULT 1")
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_dwell_tracking INTEGER DEFAULT 1")
            cursor.execute("ALTER TABLE camera_settings ADD COLUMN enable_apparel_color INTEGER DEFAULT 1")
            conn.commit()
        except Exception:
            pass

    conn.execute('''
        UPDATE camera_settings
        SET camera_source = ?, rtsp_ip = ?, rtsp_port = ?, rtsp_user = ?, rtsp_pass = ?,
            rtsp_path = ?, rtsp_url_override = ?, cooldown_minutes = ?, face_threshold = ?, gesture_threshold = ?,
            enable_machine_detection = ?, enable_dwell_tracking = ?, enable_apparel_color = ?
        WHERE id = 1
    ''', (
        camera_source, rtsp_ip.strip(), rtsp_port.strip(), rtsp_user.strip(), rtsp_pass.strip(),
        rtsp_path.strip(), rtsp_url_override.strip(), int(cooldown_minutes), float(face_threshold), float(gesture_threshold),
        int(enable_machine_detection), int(enable_dwell_tracking), int(enable_apparel_color)
    ))
    conn.commit()
    conn.close()
    return True

def log_machine_pass_event(emp_id, track_id, dwell_seconds, top_dress_color):
    conn = get_db_connection()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%Y-%m-%d %H:%M:%S')
    hour_of_day = now.hour
    status = 'HIGH_DWELL' if dwell_seconds >= 5.0 else 'PASSED'

    conn.execute('''
        INSERT INTO machine_pass_events (emp_id, person_track_id, date, timestamp, hour_of_day, dwell_seconds, top_dress_color, status_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (emp_id, track_id, date_str, time_str, hour_of_day, round(dwell_seconds, 1), top_dress_color, status))
    conn.commit()
    conn.close()

def get_machine_analytics_summary():
    conn = get_db_connection()
    today_str = date.today().strftime('%Y-%m-%d')

    # Total persons passed machine front today
    total_passed = conn.execute('SELECT COUNT(*) as count FROM machine_pass_events WHERE date = ?', (today_str,)).fetchone()['count']
    
    # High dwell (spending more time in front of machine > 5 seconds)
    high_dwell_count = conn.execute('SELECT COUNT(*) as count FROM machine_pass_events WHERE date = ? AND dwell_seconds >= 5.0', (today_str,)).fetchone()['count']
    avg_dwell_sec = conn.execute('SELECT AVG(dwell_seconds) as avg_d FROM machine_pass_events WHERE date = ?', (today_str,)).fetchone()['avg_d']
    avg_dwell_sec = round(avg_dwell_sec, 1) if avg_dwell_sec else 0.0

    # Hourly distribution for peak / low time analysis
    hourly_rows = conn.execute('''
        SELECT hour_of_day, COUNT(*) as pass_count
        FROM machine_pass_events
        WHERE date = ?
        GROUP BY hour_of_day
        ORDER BY hour_of_day ASC
    ''', (today_str,)).fetchall()

    hourly_dict = {h: 0 for h in range(24)}
    for r in hourly_rows:
        hourly_dict[r['hour_of_day']] = r['pass_count']

    # Find peak and low activity hours
    sorted_hours = sorted(hourly_dict.items(), key=lambda x: x[1], reverse=True)
    peak_hour = f"{sorted_hours[0][0]:02d}:00 - {sorted_hours[0][0]+1:02d}:00 ({sorted_hours[0][1]} passes)" if total_passed > 0 else "N/A"
    
    # Low time (during operational hours 8 AM - 8 PM)
    op_hours = [h for h in range(8, 20)]
    op_sorted = sorted([(h, hourly_dict[h]) for h in op_hours], key=lambda x: x[1])
    low_hour = f"{op_sorted[0][0]:02d}:00 - {op_sorted[0][0]+1:02d}:00 ({op_sorted[0][1]} passes)" if total_passed > 0 else "N/A"

    # Apparel dress colors distribution (top worn colors)
    color_rows = conn.execute('''
        SELECT top_dress_color, COUNT(*) as color_count
        FROM machine_pass_events
        WHERE date = ? AND top_dress_color != 'Unknown'
        GROUP BY top_dress_color
        ORDER BY color_count DESC
        LIMIT 6
    ''', (today_str,)).fetchall()

    colors_data = [dict(r) for r in color_rows]

    conn.close()

    return {
        'total_passed_today': total_passed,
        'high_dwell_count': high_dwell_count,
        'avg_dwell_sec': avg_dwell_sec,
        'peak_hour': peak_hour,
        'low_hour': low_hour,
        'hourly_chart_labels': [f"{h:02d}:00" for h in range(24)],
        'hourly_chart_data': [hourly_dict[h] for h in range(24)],
        'color_labels': [c['top_dress_color'] for c in colors_data] or ['Blue', 'Black', 'White', 'Red'],
        'color_counts': [c['color_count'] for c in colors_data] or [0, 0, 0, 0]
    }
