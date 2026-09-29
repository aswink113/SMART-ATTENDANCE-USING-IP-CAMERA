// Employee Management, DP Cropper & Camera Snapshots Controller

let webcamStream = null;
let manageWebcamStream = null;
let currentSnapIndex = 1;
let currentManageSnapIndex = 1;
const snappedPhotos = { 1: null, 2: null, 3: null };
const manageSnappedPhotos = { 1: null, 2: null, 3: null };

let currentCropper = null;
let currentCropEmpId = null;
let currentCropEmpName = null;
let currentEditingEmp = null;

// ----------------- View Mode Switcher (Grid / List) -----------------
function switchEmployeeView(mode) {
    const gridView = document.getElementById('employees-grid-view');
    const listView = document.getElementById('employees-list-view');
    const btnGrid = document.getElementById('btn-view-grid');
    const btnList = document.getElementById('btn-view-list');

    if (!gridView || !listView) return;

    if (mode === 'list') {
        gridView.style.display = 'none';
        listView.style.display = 'block';
        btnList.classList.add('active');
        btnGrid.classList.remove('active');
        localStorage.setItem('employee_view_mode', 'list');
    } else {
        listView.style.display = 'none';
        gridView.style.display = 'grid';
        btnGrid.classList.add('active');
        btnList.classList.remove('active');
        localStorage.setItem('employee_view_mode', 'grid');
    }
}

// Restore saved view mode on load
document.addEventListener('DOMContentLoaded', () => {
    const savedMode = localStorage.getItem('employee_view_mode');
    if (savedMode === 'list') {
        switchEmployeeView('list');
    }
});

// ----------------- Live Search / Filter -----------------
function filterEmployees() {
    const query = (document.getElementById('employee-search-input')?.value || '').toLowerCase().trim();
    const items = document.querySelectorAll('.employee-item');
    let visibleCount = 0;

    items.forEach(item => {
        const searchData = item.getAttribute('data-search') || '';
        if (!query || searchData.includes(query)) {
            item.style.display = item.tagName === 'TR' ? '' : 'flex';
            visibleCount++;
        } else {
            item.style.display = 'none';
        }
    });

    const countDisplay = document.getElementById('employee-count-display');
    if (countDisplay) countDisplay.textContent = visibleCount;
}

// ----------------- Add Employee Modal -----------------
function openAddEmployeeModal() {
    document.getElementById('add-employee-modal').classList.add('show');
}

function closeAddEmployeeModal() {
    document.getElementById('add-employee-modal').classList.remove('show');
    stopWebcamStream();
}

function switchPhotoSource(mode) {
    const fileTab = document.getElementById('tab-file-upload');
    const webcamTab = document.getElementById('tab-webcam-snap');
    const fileMode = document.getElementById('file-upload-mode');
    const webcamMode = document.getElementById('webcam-snap-mode');

    if (mode === 'file') {
        fileTab.classList.add('active');
        webcamTab.classList.remove('active');
        fileMode.style.display = 'block';
        webcamMode.style.display = 'none';
        stopWebcamStream();
    } else {
        webcamTab.classList.add('active');
        fileTab.classList.remove('active');
        webcamMode.style.display = 'block';
        fileMode.style.display = 'none';
        startWebcamStream();
    }
}

function handleFileSelect(event) {
    const files = event.target.files;
    const previewGrid = document.getElementById('file-preview-grid');
    if (!previewGrid) return;
    previewGrid.innerHTML = '';

    if (files.length > 3) {
        alert('Please select up to 3 photos of the employee.');
    }

    Array.from(files).slice(0, 3).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'file-preview-item';
            previewGrid.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

async function startWebcamStream() {
    try {
        const video = document.getElementById('webcam-video');
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('🔒 Camera Access Blocked by Browser:\n\nChrome & Edge disable the camera API on network IP addresses (http://192.168.x.x).\n\n👉 Please open the web dashboard using:\nhttp://localhost:5000\nor\nhttp://127.0.0.1:5000\n\nThis will immediately prompt for camera permission!');
            return;
        }
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = webcamStream;
    } catch (err) {
        alert('Unable to access webcam: ' + err.message + '\n\nPlease ensure your camera privacy shutter is open and browser permissions are allowed.');
    }
}

function stopWebcamStream() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

function captureSnapshot() {
    const video = document.getElementById('webcam-video');
    if (!video || !webcamStream) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Data = canvas.toDataURL('image/jpeg', 0.9);
    const slotIdx = currentSnapIndex;
    snappedPhotos[slotIdx] = base64Data;

    const imgEl = document.getElementById(`snap-img-${slotIdx}`);
    const hiddenInput = document.getElementById(`hidden_photo_${slotIdx}`);
    if (imgEl && hiddenInput) {
        imgEl.src = base64Data;
        imgEl.style.display = 'block';
        hiddenInput.value = base64Data;
    }

    currentSnapIndex = (currentSnapIndex % 3) + 1;
    const badge = document.getElementById('snap-counter-badge');
    if (badge) badge.textContent = currentSnapIndex;
}

async function submitAddEmployee(e) {
    e.preventDefault();
    const form = document.getElementById('add-employee-form');
    const formData = new FormData(form);
    const submitBtn = document.getElementById('submit-emp-btn');

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i><span>Training Face Recognition...</span>';
    if (window.lucide) lucide.createIcons();

    try {
        const res = await fetch('/api/employee/add', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            showToast(data.message, 'success');
            closeAddEmployeeModal();
            setTimeout(() => window.location.reload(), 1200);
        } else {
            alert('❌ Registration Failed: ' + data.message);
        }
    } catch (err) {
        alert('Server error registering employee: ' + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i data-lucide="check"></i><span>Register & Train Face AI</span>';
        if (window.lucide) lucide.createIcons();
    }
}

// ----------------- Manage / Update Photos Modal -----------------
function openManagePhotosModal(empId, empName) {
    document.getElementById('manage_photos_emp_id').value = empId;
    document.getElementById('manage-photos-emp-name').textContent = `${empName} (${empId})`;
    document.getElementById('manage-photos-modal').classList.add('show');
}

function closeManagePhotosModal() {
    document.getElementById('manage-photos-modal').classList.remove('show');
    stopManageWebcamStream();
}

function switchManagePhotoSource(mode) {
    const fileTab = document.getElementById('tab-manage-file');
    const webcamTab = document.getElementById('tab-manage-webcam');
    const fileMode = document.getElementById('manage-file-mode');
    const webcamMode = document.getElementById('manage-webcam-mode');

    if (mode === 'file') {
        fileTab.classList.add('active');
        webcamTab.classList.remove('active');
        fileMode.style.display = 'block';
        webcamMode.style.display = 'none';
        stopManageWebcamStream();
    } else {
        webcamTab.classList.add('active');
        fileTab.classList.remove('active');
        webcamMode.style.display = 'block';
        fileMode.style.display = 'none';
        startManageWebcamStream();
    }
}

function handleManageFileSelect(event) {
    const files = event.target.files;
    const previewGrid = document.getElementById('manage-file-preview-grid');
    if (!previewGrid) return;
    previewGrid.innerHTML = '';

    Array.from(files).slice(0, 3).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'file-preview-item';
            previewGrid.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

async function startManageWebcamStream() {
    try {
        const video = document.getElementById('manage-webcam-video');
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('🔒 Camera Access Blocked by Browser:\n\nPlease open the dashboard via http://localhost:5000 or http://127.0.0.1:5000 in your browser to enable webcam captures!');
            return;
        }
        manageWebcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = manageWebcamStream;
    } catch (err) {
        alert('Unable to access webcam: ' + err.message);
    }
}

function stopManageWebcamStream() {
    if (manageWebcamStream) {
        manageWebcamStream.getTracks().forEach(track => track.stop());
        manageWebcamStream = null;
    }
}

function captureManageSnapshot() {
    const video = document.getElementById('manage-webcam-video');
    if (!video || !manageWebcamStream) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Data = canvas.toDataURL('image/jpeg', 0.9);
    const slotIdx = currentManageSnapIndex;
    manageSnappedPhotos[slotIdx] = base64Data;

    const imgEl = document.getElementById(`manage-snap-img-${slotIdx}`);
    const hiddenInput = document.getElementById(`manage_hidden_photo_${slotIdx}`);
    if (imgEl && hiddenInput) {
        imgEl.src = base64Data;
        imgEl.style.display = 'block';
        hiddenInput.value = base64Data;
    }

    currentManageSnapIndex = (currentManageSnapIndex % 3) + 1;
    const badge = document.getElementById('manage-snap-counter-badge');
    if (badge) badge.textContent = currentManageSnapIndex;
}

async function submitManagePhotos(e) {
    e.preventDefault();
    const form = document.getElementById('manage-photos-form');
    const formData = new FormData(form);
    const submitBtn = document.getElementById('submit-manage-photos-btn');

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i><span>Retraining Face Model...</span>';
    if (window.lucide) lucide.createIcons();

    try {
        const res = await fetch('/api/employee/photos/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            showToast('🎉 ' + data.message, 'success');
            closeManagePhotosModal();
            setTimeout(() => window.location.reload(), 1200);
        } else {
            alert('❌ ' + data.message);
        }
    } catch (err) {
        alert('Failed to update photos: ' + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i data-lucide="upload"></i><span>Save Photos & Retrain Model</span>';
        if (window.lucide) lucide.createIcons();
    }
}

// ----------------- Interactive Profile Picture (DP) Image Cropper -----------------
function openCropDpModal(empId, empName, initialPhotoUrl, photoList = null) {
    currentCropEmpId = empId;
    currentCropEmpName = empName;
    const titleEl = document.getElementById('cropper-emp-title');
    if (titleEl) titleEl.textContent = `${empName} (${empId})`;

    // Populate registered sample thumbnails in cropper bar
    const sampleBox = document.getElementById('cropper-sample-thumbs');
    if (sampleBox) {
        sampleBox.innerHTML = '';
        let photos = [];
        if (Array.isArray(photoList) && photoList.length > 0) {
            photos = photoList;
        } else if (currentEditingEmp && currentEditingEmp.emp_id === empId && currentEditingEmp.photos) {
            photos = currentEditingEmp.photos;
        } else if (initialPhotoUrl) {
            photos = [initialPhotoUrl];
        }

        photos.forEach(p => {
            if (!p) return;
            const cleanP = '/' + p.replace(/^[\/\\]+/, '');
            const img = document.createElement('img');
            img.src = cleanP;
            img.className = 'cropper-pick-thumb';
            img.title = 'Click to crop this photo';
            img.onclick = () => loadCropperImage(cleanP);
            sampleBox.appendChild(img);
        });
    }

    const modal = document.getElementById('crop-dp-modal');
    if (modal) modal.classList.add('show');

    const startImg = initialPhotoUrl ? ('/' + initialPhotoUrl.replace(/^[\/\\]+/, '')) : '';
    if (startImg) {
        loadCropperImage(startImg);
    }
    if (window.lucide) lucide.createIcons();
}

function closeCropDpModal() {
    const modal = document.getElementById('crop-dp-modal');
    modal.classList.remove('show');
    if (currentCropper) {
        currentCropper.destroy();
        currentCropper = null;
    }
}

function loadCropperImage(src) {
    const targetImg = document.getElementById('cropper-target-img');
    if (!targetImg) return;

    if (currentCropper) {
        currentCropper.destroy();
        currentCropper = null;
    }

    targetImg.src = src;

    // Initialize Cropper.js when image loads
    targetImg.onload = () => {
        currentCropper = new Cropper(targetImg, {
            aspectRatio: 1,
            viewMode: 1,
            dragMode: 'move',
            autoCropArea: 0.85,
            restore: false,
            guides: true,
            center: true,
            highlight: false,
            cropBoxMovable: true,
            cropBoxResizable: true,
            toggleDragModeOnDblclick: false,
            preview: '.cropper-preview-container',
            ready() {
                if (window.lucide) lucide.createIcons();
            }
        });
    };
}

function handleCropperCustomFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        loadCropperImage(event.target.result);
    };
    reader.readAsDataURL(file);
}

function cropperZoom(delta) {
    if (currentCropper) currentCropper.zoom(delta);
}

function cropperRotate(deg) {
    if (currentCropper) currentCropper.rotate(deg);
}

function cropperReset() {
    if (currentCropper) currentCropper.reset();
}

async function saveCroppedDp() {
    if (!currentCropper || !currentCropEmpId) {
        alert('Please select and crop an image first.');
        return;
    }

    const btn = document.getElementById('btn-save-crop-dp');
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="spin"></i><span>Saving DP...</span>';
    if (window.lucide) lucide.createIcons();

    const canvas = currentCropper.getCroppedCanvas({
        width: 320,
        height: 320,
        imageSmoothingEnabled: true,
        imageSmoothingQuality: 'high'
    });

    const base64Data = canvas.toDataURL('image/jpeg', 0.9);

    try {
        const res = await fetch('/api/employee/dp/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emp_id: currentCropEmpId,
                image: base64Data
            })
        });
        const data = await res.json();

        if (data.success) {
            showToast('🎉 ' + data.message, 'success');
            closeCropDpModal();
            setTimeout(() => window.location.reload(), 800);
        } else {
            alert('❌ Failed to save DP: ' + data.message);
        }
    } catch (err) {
        alert('Server error saving DP: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="check"></i><span>Save & Set as Profile Picture (DP)</span>';
        if (window.lucide) lucide.createIcons();
    }
}

// ----------------- Edit Modal with DP Preview -----------------
function openEditModal(empOrId) {
    let emp = empOrId;
    if (typeof empOrId === 'string') {
        if (window.EMPLOYEES_DATA && Array.isArray(window.EMPLOYEES_DATA)) {
            emp = window.EMPLOYEES_DATA.find(e => e.emp_id === empOrId);
        }
    }
    if (!emp) {
        console.error('Employee not found:', empOrId);
        return;
    }
    currentEditingEmp = emp;
    document.getElementById('edit_emp_id').value = emp.emp_id || '';
    if (document.getElementById('edit_emp_code')) {
        document.getElementById('edit_emp_code').value = emp.emp_id || '';
    }
    document.getElementById('edit_name').value = emp.name || '';
    document.getElementById('edit_email').value = emp.email || '';
    document.getElementById('edit_phone').value = emp.phone || '';
    document.getElementById('edit_department').value = emp.department || 'General';
    document.getElementById('edit_new_password').value = '';

    document.getElementById('edit_modal_title_name').textContent = `${emp.name} (${emp.emp_id})`;
    document.getElementById('edit_modal_info_name').textContent = emp.name;
    document.getElementById('edit_modal_info_id').textContent = `ID: ${emp.emp_id} | ${emp.department}`;

    // Show DP
    const dpImg = document.getElementById('edit_modal_dp_img');
    const activeDp = emp.profile_photo || (emp.photos && emp.photos.length > 0 ? emp.photos[0] : '');
    if (activeDp) {
        dpImg.src = '/' + activeDp.replace(/^[\/\\]+/, '');
        dpImg.style.display = 'block';
    } else {
        dpImg.src = 'https://ui-avatars.com/api/?name=' + encodeURIComponent(emp.name) + '&background=6366f1&color=fff';
    }

    // Populate registered sample thumbnails in edit modal
    const photosStrip = document.getElementById('edit_modal_photos_strip');
    if (photosStrip) {
        photosStrip.innerHTML = '';
        if (emp.photos && emp.photos.length > 0) {
            emp.photos.forEach(p => {
                const cleanP = '/' + p.replace(/^[\/\\]+/, '');
                const thumbWrap = document.createElement('div');
                thumbWrap.className = 'thumb-with-actions';
                thumbWrap.innerHTML = `
                    <img src="${cleanP}" alt="Sample" class="gallery-thumb" onclick="previewImage('${cleanP}')">
                    <button type="button" class="btn-thumb-set-dp" title="Crop & Set as DP" onclick="openCropDpModal('${emp.emp_id}', '${emp.name}', '${p}')">
                        <i data-lucide="crop"></i>
                    </button>
                `;
                photosStrip.appendChild(thumbWrap);
            });
        } else {
            photosStrip.innerHTML = '<span class="text-muted" style="font-size:0.8rem;">No photos registered.</span>';
        }
    }

    document.getElementById('edit-employee-modal').classList.add('show');
    if (window.lucide) lucide.createIcons();
}

function triggerEditModalCropDp() {
    if (currentEditingEmp) {
        openCropDpModal(
            currentEditingEmp.emp_id,
            currentEditingEmp.name,
            currentEditingEmp.profile_photo || (currentEditingEmp.photos ? currentEditingEmp.photos[0] : '')
        );
    }
}

function closeEditModal() {
    document.getElementById('edit-employee-modal').classList.remove('show');
}

async function submitEditEmployee(e) {
    e.preventDefault();
    const form = document.getElementById('edit-employee-form');
    const formData = new FormData(form);

    try {
        const res = await fetch('/api/employee/edit', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            showToast('Employee updated successfully!', 'success');
            closeEditModal();
            setTimeout(() => window.location.reload(), 800);
        } else {
            alert('Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to update employee: ' + err);
    }
}

// ----------------- Delete Employee -----------------
async function deleteEmployee(empId, name) {
    if (!confirm(`Are you sure you want to delete employee "${name}" (${empId})? All photos and attendance logs will be removed.`)) {
        return;
    }

    try {
        const res = await fetch('/api/employee/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emp_id: empId })
        });
        const data = await res.json();
        if (data.success) {
            showToast(`Employee ${name} deleted successfully!`, 'info');
            setTimeout(() => window.location.reload(), 800);
        } else {
            alert('Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to delete employee: ' + err);
    }
}

// ----------------- Retrain Model Trigger -----------------
async function triggerModelRetrain() {
    const icon = document.getElementById('retrain-icon');
    if (icon) icon.classList.add('spin');

    try {
        const res = await fetch('/api/employee/retrain', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast('✅ Face AI Model Retrained: ' + data.message, 'success');
            setTimeout(() => window.location.reload(), 1000);
        } else {
            alert('Retrain note: ' + data.message);
        }
    } catch (err) {
        alert('Failed to trigger retrain: ' + err);
    } finally {
        if (icon) icon.classList.remove('spin');
    }
}
