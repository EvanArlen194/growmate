// CSS imports
import '../styles/styles.css'

import App from './pages/app.js';
import Camera from './utils/camera.js';
import { registerServiceWorker } from './utils/index.js';

// DOM
document.addEventListener('DOMContentLoaded', async () => {
  const app = new App({
    content: document.querySelector('#main-content'),
    hamburgerMenu: document.querySelector('#hamburger-menu'),
    mobileNavigation: document.querySelector('#mobile-navigation'),
    navigation: document.querySelector('#navigation'),
    sidebarMenu: document.querySelectorAll('.sidebar-menu'),
    mainMenu: document.querySelector('#main-menu')
  });

  await app.renderPage();

  await registerServiceWorker();

  window.addEventListener('hashchange', async () => {
    await app.renderPage();

    // Stop all active media
    Camera.stopAllStreams();
  });
});