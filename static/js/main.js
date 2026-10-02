// Main Frontend Controller for VisionAttend

document.addEventListener('DOMContentLoaded', () => {
    // 1. Clock Initialization
    function updateClock() {
        const clockEl = document.getElementById('live-clock');
        if (clockEl) {
            const now = new Date();
            clockEl.textContent = now.toLocaleTimeString('en-US', {
                hour12: true,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Sidebar Toggle (Mobile / Responsive)
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    // 3. Real-Time Punch Event Polling
    let lastSeenEventMsg = null;
    async function pollLiveStatus() {
        try {
            const res = await fetch('/api/live_status');
            if (!res.ok) return;
            const data = await res.json();

            // Update FPS & Camera status if elements exist
            const fpsBadge = document.getElementById('camera-fps-badge');
            if (fpsBadge && data.fps !== undefined) {
                fpsBadge.textContent = `FPS: ${data.fps}`;
            }

            const camStatusText = document.getElementById('camera-status-text');
            const statusPill = document.querySelector('.status-indicator-pill');
            if (camStatusText && data.camera_status) {
                const isOnline = data.camera_status.includes('Online');
                camStatusText.textContent = isOnline ? `AI Camera ${data.camera_status}` : `AI Camera Offline`;
                if (statusPill) {
                    statusPill.style.background = isOnline ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)';
                    statusPill.style.color = isOnline ? '#4ade80' : '#f87171';
                    statusPill.style.borderColor = isOnline ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)';
                }
            }

            // Check for new punch event
            if (data.latest_event && data.latest_event.message !== lastSeenEventMsg) {
                lastSeenEventMsg = data.latest_event.message;
                handleLivePunchEvent(data.latest_event);
            }
        } catch (err) {
            console.warn('Polling error:', err);
        }
    }

    // Poll every 1.5 seconds
    setInterval(pollLiveStatus, 1500);

    function handleLivePunchEvent(event) {
        const status = event.status || '';
        const name = event.name || 'Employee';
        const msg = event.message || '';
        const pType = event.type || 'IN';

        let toastType = 'info';
        if (status === 'IN_SUCCESS') {
            toastType = 'success';
            playAudioChime('in');
        } else if (status === 'OUT_SUCCESS') {
            toastType = 'info';
            playAudioChime('out');
        } else if (status === 'COOLDOWN') {
            toastType = 'warning';
        }

        showToast(msg, toastType);

        // If on admin dashboard, prepend to activity feed
        const feedList = document.getElementById('activity-feed-list');
        if (feedList) {
            const emptyState = feedList.querySelector('.empty-state-card');
            if (emptyState) emptyState.remove();

            const feedItem = document.createElement('div');
            feedItem.className = 'feed-item';
            feedItem.innerHTML = `
                <div class="feed-badge feed-badge-${pType.toLowerCase()}">
                    <i data-lucide="${pType === 'IN' ? 'log-in' : (pType === 'OUT' ? 'log-out' : 'clock')}"></i>
                </div>
                <div class="feed-content">
                    <div class="feed-top">
                        <strong>${name}</strong>
                        <span class="feed-time">${event.time || new Date().toLocaleTimeString()}</span>
                    </div>
                    <div class="feed-bottom">
                        <span class="badge-pill ${pType === 'IN' ? 'badge-success' : (pType === 'OUT' ? 'badge-info' : 'badge-warning')}">${pType}</span>
                        <span class="feed-dept">${event.emp_id || ''}</span>
                    </div>
                </div>
            `;
            feedList.insertBefore(feedItem, feedList.firstChild);
            if (window.lucide) lucide.createIcons();
        }
    }

    function playAudioChime(type) {
        try {
            const audio = document.getElementById(type === 'in' ? 'punch-audio-in' : 'punch-audio-out');
            if (audio) {
                audio.currentTime = 0;
                audio.play().catch(e => console.log('Audio autoplay prevented'));
            }
        } catch (e) {}
    }
});

// Global Toast Notification Helper
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconName = 'info';
    if (type === 'success') iconName = 'check-circle-2';
    if (type === 'warning') iconName = 'alert-triangle';
    if (type === 'danger') iconName = 'alert-circle';

    toast.innerHTML = `
        <i data-lucide="${iconName}"></i>
        <div style="flex:1;">${message}</div>
    `;

    container.appendChild(toast);
    if (window.lucide) lucide.createIcons();

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

// Global Password Visibility Toggle
function togglePasswordVisibility(inputId, btn) {
    const input = document.getElementById(inputId);
    if (!input) return;

    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';

    btn.innerHTML = `<i data-lucide="${isPassword ? 'eye-off' : 'eye'}"></i>`;
    btn.setAttribute('title', isPassword ? 'Hide Password' : 'Show Password');
    if (window.lucide) {
        lucide.createIcons();
    }
}

