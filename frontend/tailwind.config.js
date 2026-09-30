/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Threat level palette, matching the ThreatLevel enum in the backend.
        threat: {
          none: '#64748b',
          low: '#0891b2',
          medium: '#ca8a04',
          high: '#ea580c',
          critical: '#b91c1c',
        },
      },
    },
  },
  plugins: [],
}
