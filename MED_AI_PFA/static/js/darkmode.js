// FILE: static/js/darkmode.js (fixed for single button, improved persistence)
(function() {
    function getInitialTheme() {
        const savedTheme = localStorage.getItem('darkMode');
        if (savedTheme === 'enabled') return true;
        if (savedTheme === 'disabled') return false;
        // Fallback to system preference
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    function setDarkMode(isDark) {
        if (isDark) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        localStorage.setItem('darkMode', isDark ? 'enabled' : 'disabled');
        // Update button icon if exists
        const toggleBtn = document.getElementById('darkModeToggle');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    function initDarkMode() {
        const toggleBtn = document.getElementById('darkModeToggle');
        if (!toggleBtn) return;

        const isDark = getInitialTheme();
        setDarkMode(isDark);

        toggleBtn.addEventListener('click', function() {
            const newIsDark = !document.body.classList.contains('dark-mode');
            setDarkMode(newIsDark);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDarkMode);
    } else {
        initDarkMode();
    }
})();