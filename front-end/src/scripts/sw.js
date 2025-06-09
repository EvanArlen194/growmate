import { precacheAndRoute } from "workbox-precaching";
import { registerRoute } from "workbox-routing";
import { CacheableResponsePlugin } from "workbox-cacheable-response";
import { NetworkFirst, CacheFirst, StaleWhileRevalidate } from "workbox-strategies";
import CONFIG from "./config.js";

// Do precaching
const manifest = self.__WB_MANIFEST;
precacheAndRoute(manifest);

// Runtime caching
registerRoute(
  ({ url }) => {
    return url.origin === 'https://fonts.googleapis.com' || url.origin === 'https://fonts.gstatic.com';
  },
  new CacheFirst({
    cacheName: 'google-fonts',
  }),
);
registerRoute(
  ({ url }) => {
    return url.origin === 'https://cdnjs.cloudflare.com' || url.origin.includes('fontawesome');
  },
  new CacheFirst({
    cacheName: 'fontawesome',
  }),
);
registerRoute(
  ({ url }) => {
    return url.origin === 'https://ui-avatars.com';
  },
  new CacheFirst({
    cacheName: 'avatars-api',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
  }),
);

// Handle API

const API_URLS = [
  CONFIG.PEST_BASE_URL,
  CONFIG.DISEASE_BASE_URL,
  CONFIG.RECOM_BASE_URL,
]

API_URLS.forEach((baseUrl) => {
  const origin = new URL(baseUrl).origin;

  // Non-image request
  registerRoute(
    ({ request, url }) => url.origin === origin && request.destination !== 'image',
    new NetworkFirst({ cacheName: `${origin}-api` })
  );

  // Image request
  registerRoute(
    ({ request, url }) => url.origin === origin && request.destination === 'image',
    new StaleWhileRevalidate({ cacheName: `${origin}-images` })
  );
});

registerRoute(
  ({ url }) => url.origin === 'https://cdn.jsdelivr.net',
  new CacheFirst({
    cacheName: 'cdn-jsdelivr-assets',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
  })
);

registerRoute(
  ({ url, request }) => {
    return request.destination === 'image' && url.origin === self.location.origin;
  },
  new CacheFirst({
    cacheName: 'local-images',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
  })
);

registerRoute(
  ({ url }) => url.pathname.startsWith('/images/'),
  new CacheFirst({
    cacheName: 'local-images',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
  })
);
