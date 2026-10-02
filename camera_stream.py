import cv2
import threading
import time
import os
import socket
import urllib.parse
from datetime import datetime
import numpy as np
from config import CAPTURES_DIR
from database import get_camera_settings, process_gesture_punch, get_employee_by_id, log_machine_pass_event
from face_engine import face_engine
from gesture_engine import gesture_engine, analyze_top_apparel_color

class CameraStreamManager:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.ai_thread = None
        self.lock = threading.Lock()
        
        self.latest_raw_frame = None
        self.latest_detections = {
            'faces': [],
            'hands': [],
            'is_hand_raised': False,
            'timestamp': 0,
            'dwell_events': []
        }
        
        self.fps = 0.0
        self.ai_fps = 0.0
        self.latest_punch_event = None
        self.latest_event_time = 0
        self.camera_status = "Disconnected"
        self.camera_source_str = "Webcam 0"
        
        # Debounce tracking for hand gestures
        self.last_punch_trigger_time = {} # emp_id -> timestamp
        # Active dwell tracking for machine front passage: emp_id/track_id -> {start_time, last_seen, max_dwell, logged, dress_color}
        self.active_person_tracks = {}
        self.track_counter = 0
        self.is_paused = False
        self.is_client_streaming = False
        self.last_client_frame_time = 0

        # Spatial Heatmap Dwell Accumulation Matrix (64 x 48 grid) & Display Toggle
        self.heatmap_grid = np.zeros((48, 64), dtype=np.float32)
        self.enable_heatmap = True

        # Custom user-marked target location (normalized x, y between 0.0 - 1.0)
        self.marked_location = None  # tuple (x_norm, y_norm, label)
        # Hourly occupancy tracking for the marked location: hour_str "00:00".."23:00" -> count of detections
        self.marked_location_hourly_counts = {f"{h:02d}:00": 0 for h in range(24)}

        self.start()

    def get_source_from_settings(self):
        settings = get_camera_settings()
        source_type = str(settings.get('camera_source', '0')).strip()
        url_override = settings.get('rtsp_url_override', '').strip()

        if url_override:
            self.camera_source_str = f"Custom: {url_override}"
            return url_override

        if source_type in ['0', '1', '2']:
            self.camera_source_str = f"Local Webcam ({source_type})"
            return int(source_type)
        else:
            # Network IP Camera (RTSP / HTTP)
            ip = settings.get('rtsp_ip', '').strip()
            port = str(settings.get('rtsp_port', '554')).strip() or '554'
            user = settings.get('rtsp_user', '').strip()
            password = settings.get('rtsp_pass', '').strip()
            path = settings.get('rtsp_path', '').strip()
            if path and not path.startswith('/'):
                path = '/' + path

            # Handle direct full URLs in the IP field
            if ip.startswith('rtsp://') or ip.startswith('http://') or ip.startswith('https://'):
                self.camera_source_str = f"IP Camera ({ip})"
                return ip

            if not ip:
                self.camera_source_str = "Local Webcam (0)"
                return 0

            user_enc = urllib.parse.quote(user, safe='') if user else ''
            pass_enc = urllib.parse.quote(password, safe='') if password else ''
            auth = f"{user_enc}:{pass_enc}@" if (user_enc and pass_enc) else (f"{user_enc}@" if user_enc else "")

            host_port = f"{ip}:{port}"
            rtsp_url = f"rtsp://{auth}{host_port}{path}"
            self.camera_source_str = f"RTSP: {ip}:{port}{path}"
            return rtsp_url

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        
        # 1. Start Hardware / RTSP Capture Thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # 2. Start Asynchronous Dedicated AI Worker Thread
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.ai_thread.start()

    def restart(self):
        print("Restarting camera stream with updated settings...")
        self.is_running = False
        
        # Release capture device immediately to unblock pending read()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)

        self.is_paused = False
        self.is_client_streaming = False
        self.start()

    def _open_camera(self, source):
        if isinstance(source, int):
            # Try DirectShow, Media Foundation, and Default backend
            for api_backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(source, api_backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, test_f = cap.read()
                        if ret and test_f is not None:
                            print(f"Connected to Camera {source} with backend {api_backend} at 640x480")
                            return cap
                        cap.release()
                except Exception as e:
                    print(f"Backend {api_backend} failed: {e}")
            
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        else:
            # Network IP Camera (RTSP / HTTP)
            source_str = str(source).strip()
            print(f"Connecting to Network IP Camera: {source_str}")

            # 1. Try FFMPEG with TCP transport
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;3000000"
                cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_f = cap.read()
                    if ret and test_f is not None:
                        print(f"[OK] Connected to IP Camera via FFMPEG (TCP): {source_str}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"FFMPEG TCP attempt error: {e}")

            # 2. Try FFMPEG with UDP transport fallback
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"
                cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_f = cap.read()
                    if ret and test_f is not None:
                        print(f"[OK] Connected to IP Camera via FFMPEG (UDP): {source_str}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"FFMPEG UDP attempt error: {e}")

            # 3. Try CAP_ANY fallback
            try:
                cap = cv2.VideoCapture(source_str, cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_f = cap.read()
                    if ret and test_f is not None:
                        print(f"[OK] Connected to IP Camera via CAP_ANY: {source_str}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"CAP_ANY attempt error: {e}")

            print(f"[ERROR] Failed to connect to IP Camera stream: {source_str}")
            return None

    @staticmethod
    def test_ip_camera(ip='', port='554', user='', password='', path='', url_override=None):
        """Standalone diagnostic probe to test RTSP/HTTP camera connections."""
        if url_override and str(url_override).strip():
            candidate_urls = [str(url_override).strip()]
            test_host = "127.0.0.1"
            test_port = 554
            try:
                parsed = urllib.parse.urlparse(str(url_override).strip())
                test_host = parsed.hostname or test_host
                test_port = parsed.port or (554 if parsed.scheme == 'rtsp' else 80)
            except Exception:
                pass
        else:
            ip = str(ip or '').strip()
            if not ip:
                return {'success': False, 'message': 'Please provide an IP address or hostname.'}

            if ip.startswith('rtsp://') or ip.startswith('http://') or ip.startswith('https://'):
                candidate_urls = [ip]
                test_host = "127.0.0.1"
                test_port = 554
                try:
                    parsed = urllib.parse.urlparse(ip)
                    test_host = parsed.hostname or test_host
                    test_port = parsed.port or 554
                except Exception:
                    pass
            else:
                test_host = ip
                try:
                    test_port = int(str(port).strip()) if port and str(port).strip() else 554
                except (ValueError, TypeError):
                    test_port = 554

                user_str = str(user or '').strip()
                pass_str = str(password or '').strip()
                user_enc = urllib.parse.quote(user_str, safe='') if user_str else ''
                pass_enc = urllib.parse.quote(pass_str, safe='') if pass_str else ''
                auth = f"{user_enc}:{pass_enc}@" if (user_enc and pass_enc) else (f"{user_enc}@" if user_enc else "")
                
                clean_path = str(path or '').strip()
                if clean_path and not clean_path.startswith('/'):
                    clean_path = '/' + clean_path

                host_port = f"{ip}:{test_port}"

                # Paths to probe in order (Hikvision, Dahua/CP PLUS, Reolink, ONVIF, generic)
                paths_to_try = [
                    clean_path,
                    '/Streaming/Channels/101',
                    '/Streaming/Channels/102',
                    '/Streaming/channels/101',
                    '/Streaming/channels/102',
                    '/ISAPI/Streaming/channels/101',
                    '/ISAPI/Streaming/channels/102',
                    '/h264Preview_01_main',
                    '/cam/realmonitor?channel=1&subtype=0',
                    '/cam/realmonitor?channel=1&subtype=1',
                    '/live/ch0',
                    '/stream1',
                    '/stream',
                    '/onvif1',
                    '/h264',
                    ''
                ]
                # Filter out empty or duplicate entries while preserving order
                unique_paths = []
                for p in paths_to_try:
                    if p not in unique_paths:
                        unique_paths.append(p)

                candidate_urls = []
                for p in unique_paths:
                    candidate_urls.append(f"rtsp://{auth}{host_port}{p}")
                # HTTP video stream candidates
                candidate_urls.append(f"http://{auth}{host_port}/video")
                candidate_urls.append(f"http://{auth}{host_port}/mjpeg")

        # 1. Test TCP socket reachability
        socket_reachable = False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.5)
            sock_res = sock.connect_ex((test_host, test_port))
            sock.close()
            socket_reachable = (sock_res == 0)
        except Exception:
            socket_reachable = False

        if not socket_reachable:
            return {
                'success': False,
                'message': f"Cannot reach IP host '{test_host}' on port {test_port}. Please check: 1) Camera is powered ON, 2) Both computer & camera are on the same local network/Wi-Fi, 3) Correct IP address is entered.",
                'socket_open': False
            }

        # 2. Try candidate RTSP/HTTP URLs
        for candidate in candidate_urls:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;2500000"
            try:
                cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        return {
                            'success': True,
                            'message': f"Connected successfully! Stream resolution: {w}x{h}",
                            'working_url': candidate,
                            'resolution': f"{w}x{h}",
                            'socket_open': True
                        }
            except Exception:
                pass

        return {
            'success': False,
            'message': f"Port {test_port} on {test_host} is reachable, but RTSP authentication or stream path failed. Please verify: 1) Camera Username & Password, 2) Stream Path (e.g. /Streaming/Channels/101 or /cam/realmonitor?channel=1&subtype=0), 3) RTSP is enabled in camera settings.",
            'socket_open': True
        }

    def pause_hardware(self):
        """Release hardware camera so browser can acquire it without device lock conflicts."""
        self.is_paused = True
        self.is_client_streaming = True
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def resume_hardware(self):
        """Resume server-side hardware camera capture."""
        self.is_paused = False
        self.is_client_streaming = False

    def feed_client_frame(self, frame_bgr):
        """Fast non-blocking ingestion of frames from browser webcam stream."""
        if frame_bgr is None:
            return
        self.camera_status = "Online (Browser Stream)"
        self.is_client_streaming = True
        self.last_client_frame_time = time.time()
        with self.lock:
            self.latest_raw_frame = frame_bgr

    def _capture_loop(self):
        """Dedicated high-speed thread for hardware camera / RTSP frames with zero-latency buffer flushing."""
        source = self.get_source_from_settings()
        self.cap = self._open_camera(source)

        fail_count = 0

        while self.is_running:
            if self.is_paused or self.is_client_streaming:
                # If client hasn't sent frames in 5 seconds, switch back
                if self.is_client_streaming and (time.time() - self.last_client_frame_time > 5.0):
                    self.is_client_streaming = False
                time.sleep(0.05)
                continue

            if not self.cap or not self.cap.isOpened():
                self.camera_status = "Connecting..."
                time.sleep(1.0)
                if not self.is_running or self.is_paused or self.is_client_streaming:
                    continue
                source = self.get_source_from_settings()
                self.cap = self._open_camera(source)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                ret = False
                frame = None
                print(f"[ERROR] Camera read exception: {e}")

            if not ret or frame is None:
                fail_count += 1
                if fail_count > 15:
                    self.camera_status = "Camera Offline"
                    time.sleep(1.0)
                    if self.cap:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                    if not self.is_running or self.is_paused:
                        continue
                    source = self.get_source_from_settings()
                    self.cap = self._open_camera(source)
                    fail_count = 0
                time.sleep(0.03)
                continue

            fail_count = 0
            self.camera_status = "Online"

            with self.lock:
                self.latest_raw_frame = frame

            time.sleep(0.001) # Low yield sleep for minimal CPU overhead and max throughput

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.camera_status = "Stopped"

    def _ai_worker_loop(self):
        """Asynchronous worker that runs face detection and gesture recognition at peak speed."""
        ai_frame_count = 0
        ai_start_time = time.time()

        while self.is_running:
            try:
                raw_frame = None
                with self.lock:
                    if self.latest_raw_frame is not None:
                        raw_frame = self.latest_raw_frame.copy()

                if raw_frame is None:
                    time.sleep(0.03)
                    continue

                # Downscale frame for fast AI inference (~480px width)
                h, w = raw_frame.shape[:2]
                target_w = 480
                scale = 1.0
                if w > target_w:
                    scale = target_w / float(w)
                    proc_frame = cv2.resize(raw_frame, (target_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_frame = raw_frame

                settings = get_camera_settings()
                face_thresh = settings.get('face_threshold', 0.40)
                gesture_thresh = settings.get('gesture_threshold', 0.50)
                enable_machine = settings.get('enable_machine_detection', 1)
                enable_dwell = settings.get('enable_dwell_tracking', 1)
                enable_apparel = settings.get('enable_apparel_color', 1)

                # 1. Face Detection & Machine Front Passage Tracking
                detected_faces = face_engine.detect_faces(proc_frame)
                recognized_faces = []
                now_ts = time.time()
                current_seen_keys = set()

                # Extract primary face bbox if present to enforce body-aligned hand raise
                primary_face_bbox = None
                if len(detected_faces) > 0:
                    best_f = max(detected_faces, key=lambda f: float(f[2] * f[3]))
                    primary_face_bbox = (float(best_f[0]), float(best_f[1]), float(best_f[2]), float(best_f[3]))

                # 2. Gesture Detection on scaled frame with anatomical body alignment check
                is_hand_raised, detected_hands = gesture_engine.detect_gesture(proc_frame, threshold=gesture_thresh, face_bbox=primary_face_bbox)
                
                # Rescale hand bounding boxes back to full coordinates
                scaled_hands = []
                for hand in detected_hands:
                    if hand.get('bbox'):
                        hx, hy, hw, hh = hand['bbox']
                        scaled_hands.append({
                            'bbox': (int(hx / scale), int(hy / scale), int(hw / scale), int(hh / scale)),
                            'gesture': hand['gesture'],
                            'score': hand['score'],
                            'is_raised': hand['is_raised']
                        })

                # Face recognition spatial cache to avoid redundant neural network inference on consecutive frames
                if not hasattr(self, '_face_recog_cache'):
                    self._face_recog_cache = []

                new_cache = []

                for face_data in detected_faces:
                    # Rescale face data to match original frame
                    rescaled_face = face_data.copy()
                    rescaled_face[0] /= scale
                    rescaled_face[1] /= scale
                    rescaled_face[2] /= scale
                    rescaled_face[3] /= scale
                    if len(rescaled_face) >= 14: # Landmarks
                        for k in range(4, 14, 2):
                            rescaled_face[k] /= scale
                            rescaled_face[k+1] /= scale

                    fx, fy, fw, fh = int(rescaled_face[0]), int(rescaled_face[1]), int(rescaled_face[2]), int(rescaled_face[3])

                    # Spatial IoU match against face recognition cache
                    cached_match = None
                    for c_entry in self._face_recog_cache:
                        if now_ts - c_entry['ts'] < 0.30: # Cache valid for 300ms (~9 frames)
                            cx, cy, cw, ch = c_entry['bbox']
                            ix1, iy1 = max(fx, cx), max(fy, cy)
                            ix2, iy2 = min(fx + fw, cx + cw), min(fy + fh, cy + ch)
                            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                            inter = iw * ih
                            union = (fw * fh) + (cw * ch) - inter
                            iou = inter / float(union) if union > 0 else 0.0
                            if iou >= 0.55:
                                cached_match = c_entry
                                break

                    if cached_match:
                        emp_id = cached_match['emp_id']
                        confidence = cached_match['confidence']
                    else:
                        emp_id, confidence, (fx, fy, fw, fh) = face_engine.recognize_face(raw_frame, rescaled_face, threshold=face_thresh)
                        cached_match = {'bbox': (fx, fy, fw, fh), 'emp_id': emp_id, 'confidence': confidence, 'ts': now_ts}

                    new_cache.append(cached_match)
                    
                    emp_name = "Unregistered"
                    dept_name = ""
                    if emp_id != "Unknown":
                        emp_info = get_employee_by_id(emp_id)
                        emp_name = emp_info['name'] if emp_info else emp_id
                        dept_name = emp_info.get('department', '') if emp_info else ''

                        # Check Gesture Punch Trigger
                        if is_hand_raised:
                            last_trig = self.last_punch_trigger_time.get(emp_id, 0)
                            if now_ts - last_trig > 2.0:
                                self.last_punch_trigger_time[emp_id] = now_ts
                                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                                snapshot_filename = f"punch_{emp_id}_{timestamp_str}.jpg"
                                snapshot_full_path = os.path.join(CAPTURES_DIR, snapshot_filename)
                                snapshot_rel_path = f"captures/{snapshot_filename}"
                                
                                try:
                                    cv2.imwrite(snapshot_full_path, raw_frame)
                                except Exception:
                                    pass

                                punch_res = process_gesture_punch(
                                    emp_id=emp_id,
                                    snapshot_path=snapshot_rel_path,
                                    confidence=confidence
                                )
                                self.latest_punch_event = punch_res
                                self.latest_event_time = time.time()

                    # Apparel Top Dress Color Analysis (if enabled)
                    top_color = "Unknown"
                    if enable_apparel:
                        top_color = analyze_top_apparel_color(raw_frame, face_bbox=(fx, fy, fw, fh))

                    # Machine Front Dwell Tracking (if enabled)
                    # Compute bounding box for person (fx, fy, fw, fh)
                    curr_box = (fx, fy, fw, fh)
                    
                    # Try to associate face/person with existing active track using identity OR spatial IoU overlap
                    matched_key = None

                    # 1. First priority match: match by registered employee identity (if recognized)
                    if emp_id != "Unknown" and emp_id in self.active_person_tracks:
                        matched_key = emp_id

                    # 2. Second priority match: Spatial IoU / centroid proximity match for active tracks
                    if matched_key is None:
                        best_iou = 0.0
                        for tkey, tr in self.active_person_tracks.items():
                            prev_box = tr.get('bbox')
                            if prev_box:
                                # Calculate Intersection over Union (IoU) between bounding boxes
                                px, py, pw, ph = prev_box
                                ix1 = max(fx, px)
                                iy1 = max(fy, py)
                                ix2 = min(fx + fw, px + pw)
                                iy2 = min(fy + fh, py + ph)
                                iw = max(0, ix2 - ix1)
                                ih = max(0, iy2 - iy1)
                                inter_area = iw * ih
                                box1_area = fw * fh
                                box2_area = pw * ph
                                union_area = box1_area + box2_area - inter_area
                                iou = inter_area / float(union_area) if union_area > 0 else 0.0

                                if iou > 0.25 and iou > best_iou:
                                    best_iou = iou
                                    matched_key = tkey

                    # 3. Create or update track
                    if matched_key is None:
                        # New track key
                        track_key = emp_id if emp_id != "Unknown" else f"anon_tr_{self.track_counter + 1}"
                        self.track_counter += 1
                        self.active_person_tracks[track_key] = {
                            'track_id': self.track_counter,
                            'emp_id': emp_id,
                            'start_time': now_ts,
                            'last_seen': now_ts,
                            'dwell_seconds': 0.0,
                            'logged': False,
                            'dress_color': top_color,
                            'bbox': curr_box
                        }
                    else:
                        track_key = matched_key
                        tr = self.active_person_tracks[matched_key]
                        
                        # Update identity if previously anonymous but now recognized
                        if tr.get('emp_id', 'Unknown') == "Unknown" and emp_id != "Unknown":
                            # Upgrade track key if emp_id is not already taken
                            if emp_id not in self.active_person_tracks:
                                self.active_person_tracks[emp_id] = tr
                                del self.active_person_tracks[track_key]
                                track_key = emp_id
                            tr['emp_id'] = emp_id

                        tr['last_seen'] = now_ts
                        tr['dwell_seconds'] = now_ts - tr['start_time']
                        tr['bbox'] = curr_box
                        if top_color != "Unknown":
                            tr['dress_color'] = top_color

                    current_seen_keys.add(track_key)

                    # Accumulate spatial dwell position density in heatmap grid covering full body region
                    # Estimate full body bounding box from face position:
                    # Head height approx 1/7 to 1/8 of total body height (approx 4x to 5x face height down, 1.8x face width across)
                    body_x1 = max(0, int(fx - 0.4 * fw))
                    body_y1 = max(0, int(fy))
                    body_x2 = min(w, int(fx + 1.4 * fw))
                    body_y2 = min(h, int(fy + 4.5 * fh))

                    gx1 = int(np.clip((body_x1 / float(w)) * 64.0, 0, 63))
                    gy1 = int(np.clip((body_y1 / float(h)) * 48.0, 0, 47))
                    gx2 = int(np.clip((body_x2 / float(w)) * 64.0, 0, 63))
                    gy2 = int(np.clip((body_y2 / float(h)) * 48.0, 0, 47))

                    # Increment dwell density across full body grid region
                    for dy in range(gy1, gy2 + 1):
                        for dx in range(gx1, gx2 + 1):
                            self.heatmap_grid[dy, dx] += 0.02

                    # Check proximity to user custom marked location
                    if self.marked_location:
                        mx_norm, my_norm, _ = self.marked_location
                        # Proximity radius check (approx 18% normalized distance)
                        dist = np.hypot(person_cx_norm - mx_norm, person_cy_norm - my_norm)
                        if dist <= 0.18:
                            current_hour = datetime.now().strftime('%H:00')
                            if current_hour in self.marked_location_hourly_counts:
                                self.marked_location_hourly_counts[current_hour] += 1

                    recognized_faces.append({
                        'emp_id': emp_id,
                        'emp_name': emp_name,
                        'dept_name': dept_name,
                        'confidence': confidence,
                        'bbox': (fx, fy, fw, fh),
                        'dress_color': top_color,
                        'dwell_seconds': round(self.active_person_tracks[track_key]['dwell_seconds'], 1) if (enable_machine and enable_dwell and track_key in self.active_person_tracks) else 0.0
                    })

                # Cleanup expired tracks & log completed machine front passage events
                expired_keys = []
                if enable_machine and enable_dwell:
                    for tkey, tr in self.active_person_tracks.items():
                        if now_ts - tr['last_seen'] > 4.0: # Person left front of camera (4.0s tolerance for momentary face occlusion/turning away)
                            expired_keys.append(tkey)
                            dwell_duration = tr['last_seen'] - tr['start_time']
                            if dwell_duration >= 1.0 and not tr['logged']:
                                emp_label = tr.get('emp_id', 'Unknown')
                                if emp_label == 'Unknown' and tkey.startswith('anon_'):
                                    emp_label = 'Unknown'
                                log_machine_pass_event(emp_label, tr['track_id'], dwell_duration, tr['dress_color'])
                                tr['logged'] = True

                    for ekey in expired_keys:
                        del self.active_person_tracks[ekey]

                # Update cached detections
                with self.lock:
                    self.latest_detections = {
                        'faces': recognized_faces,
                        'hands': scaled_hands,
                        'is_hand_raised': is_hand_raised,
                        'timestamp': time.time()
                    }

                # Calculate AI inference FPS
                ai_frame_count += 1
                ai_elapsed = time.time() - ai_start_time
                if ai_elapsed >= 1.0:
                    self.ai_fps = round(ai_frame_count / ai_elapsed, 1)
                    ai_frame_count = 0
                    ai_start_time = time.time()
            except Exception as e:
                print(f"AI Worker loop exception: {e}")

            time.sleep(0.01)

    def _draw_hud(self, frame):
        """Draws HUD overlays, face boxes, hand markers, heatmap hot zones, and alerts on a frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 0. Render Thermal Heatmap Density Overlay for High-Dwell Hot Zones (if enabled)
        max_density = float(np.max(self.heatmap_grid))
        if self.enable_heatmap and max_density > 0.5:
            # Resize grid to frame resolution using fast LINEAR interpolation
            grid_resized = cv2.resize(self.heatmap_grid, (w, h), interpolation=cv2.INTER_LINEAR)
            grid_norm = np.uint8(np.clip(grid_resized / max_density * 255.0, 0, 255))
            
            mask = grid_norm > 25
            if np.any(mask):
                heatmap_color = cv2.applyColorMap(grid_norm, cv2.COLORMAP_JET)
                # Fast in-place alpha blending on masked region only
                annotated[mask] = cv2.addWeighted(annotated, 0.65, heatmap_color, 0.35, 0)[mask]

            # Find peak dwell coordinate position
            max_pos = np.unravel_index(np.argmax(self.heatmap_grid), self.heatmap_grid.shape)
            peak_y = int((max_pos[0] + 0.5) / 48.0 * h)
            peak_x = int((max_pos[1] + 0.5) / 64.0 * w)

            # Mark the position where people spend most time
            cv2.drawMarker(annotated, (peak_x, peak_y), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
            cv2.circle(annotated, (peak_x, peak_y), 18, (0, 0, 255), 2)
            cv2.circle(annotated, (peak_x, peak_y), 32, (0, 255, 255), 1)

            # Floating Hot Zone Tag
            tag_text = "🔥 MOST FREQUENT / HIGHEST DWELL POSITION"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            tx = max(10, min(w - tw - 20, peak_x - int(tw / 2)))
            ty = max(45, peak_y - 25)
            cv2.rectangle(annotated, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (15, 23, 42), -1)
            cv2.rectangle(annotated, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 255), 1)
            cv2.putText(annotated, tag_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2)

        # Draw User Custom Marked Location Target Pin if set
        if self.marked_location:
            mx_norm, my_norm, m_label = self.marked_location
            m_px = int(mx_norm * w)
            m_py = int(my_norm * h)
            # Neon Cyan Pin Marker
            cv2.circle(annotated, (m_px, m_py), 12, (255, 255, 0), -1)
            cv2.circle(annotated, (m_px, m_py), 20, (255, 255, 0), 2)
            cv2.drawMarker(annotated, (m_px, m_py), (0, 0, 0), cv2.MARKER_CROSS, 12, 2)
            lbl = f"📍 {m_label}"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            lx = max(10, min(w - lw - 10, m_px - int(lw / 2)))
            ly = max(40, m_py - 24)
            cv2.rectangle(annotated, (lx - 4, ly - lh - 4), (lx + lw + 4, ly + 4), (6, 182, 212), -1)
            cv2.putText(annotated, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        with self.lock:
            detections = self.latest_detections.copy()
            punch_event = self.latest_punch_event
            punch_event_time = self.latest_event_time
            disp_fps = self.fps

        is_hand_raised = detections.get('is_hand_raised', False)

        # 1. Draw Hands
        for hand in detections.get('hands', []):
            hx, hy, hw, hh = hand['bbox']
            color = (0, 255, 128) if hand['is_raised'] else (200, 200, 200)
            cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), color, 2)
            gesture_text = f"Hand: {hand['gesture']} ({int(hand['score']*100)}%)"
            cv2.putText(annotated, gesture_text, (hx, max(20, hy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # 2. Draw Faces
        for f in detections.get('faces', []):
            fx, fy, fw, fh = f['bbox']
            emp_id = f['emp_id']
            emp_name = f['emp_name']
            dept_name = f['dept_name']
            confidence = f['confidence']

            if emp_id != "Unknown":
                color = (0, 255, 0)
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, 2)

                label_main = f"{emp_name} ({emp_id})"
                label_sub = f"Match: {confidence*100:.1f}% | {dept_name}"
                box_w = max(fw, 180)
                cv2.rectangle(annotated, (fx, max(0, fy - 36)), (fx + box_w, fy), (15, 23, 42), -1)
                cv2.putText(annotated, label_main, (fx + 4, max(14, fy - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
                cv2.putText(annotated, label_sub, (fx + 4, max(26, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128), 1)

                # Instruction Pill
                if is_hand_raised:
                    cv2.rectangle(annotated, (fx, fy + fh), (fx + fw, fy + fh + 20), (0, 200, 0), -1)
                    cv2.putText(annotated, "HAND RAISED - PUNCHING!", (fx + 4, fy + fh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
                else:
                    cv2.rectangle(annotated, (fx, fy + fh), (fx + fw, fy + fh + 20), (30, 41, 59), -1)
                    cv2.putText(annotated, "Raise hand to punch", (fx + 4, fy + fh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            else:
                color = (0, 0, 255)
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, 2)
                cv2.rectangle(annotated, (fx, max(0, fy - 20)), (fx + 110, fy), (0, 0, 180), -1)
                cv2.putText(annotated, "Unregistered", (fx + 4, max(14, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # 3. Top System HUD Banner
        cv2.rectangle(annotated, (0, 0), (w, 32), (15, 23, 42), -1)
        status_color = (0, 255, 128) if "Online" in self.camera_status else (0, 0, 255)
        cv2.circle(annotated, (12, 16), 5, status_color, -1)
        source_label = "Browser Webcam" if self.is_client_streaming else self.camera_source_str
        cv2.putText(annotated, f"LIVE [{source_label}] | FPS: {disp_fps}", (24, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        gesture_status_text = "✋ HAND RAISED" if is_hand_raised else "✋ Raise Hand To Punch"
        gesture_status_color = (0, 255, 128) if is_hand_raised else (180, 180, 180)
        cv2.putText(annotated, gesture_status_text, (max(10, w - 210), 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, gesture_status_color, 2 if is_hand_raised else 1)

        # 4. Punch Event Notification Banner
        if punch_event and (time.time() - punch_event_time < 4.0):
            msg = punch_event.get('message', '')
            status = punch_event.get('status', '')
            bg_color = (0, 150, 0) if status == 'IN_SUCCESS' else ((180, 100, 0) if status == 'OUT_SUCCESS' else (0, 140, 200))
            cv2.rectangle(annotated, (0, h - 36), (w, h), bg_color, -1)
            cv2.putText(annotated, f"📢 {msg}", (14, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        return annotated

    def generate_mjpeg_frames(self):
        """Streams smooth 30-FPS MJPEG video feed to the browser with zero stutter."""
        render_frame_count = 0
        render_start_time = time.time()

        while True:
            t0 = time.time()
            raw_frame = None
            with self.lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame

            if raw_frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                status_msg = f"Camera: {self.camera_status} ({self.camera_source_str})"
                cv2.putText(placeholder, status_msg, (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(placeholder, "Connecting to stream...", (30, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)
                ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:
                annotated = self._draw_hud(raw_frame)
                ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])

            render_frame_count += 1
            elapsed = time.time() - render_start_time
            if elapsed >= 1.0:
                self.fps = round(render_frame_count / elapsed, 1)
                render_frame_count = 0
                render_start_time = time.time()

            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # Precise 30 FPS dynamic cadence sleep calculation
            loop_duration = time.time() - t0
            sleep_time = max(0.001, 0.0333 - loop_duration)
            time.sleep(sleep_time)

# Global Camera Stream Singleton
camera_stream = CameraStreamManager()
