/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Amp-inspired color palette
        'amp-black': '#0a0a0b',
        'amp-charcoal': '#1a1a1d',
        'amp-steel': '#2d2d32',
        'amp-chrome': '#4a4a52',
        'amp-silver': '#8a8a95',
        'amp-cream': '#f5f0e6',
        'amp-amber': '#d4a84b',
        'amp-orange': '#e65c00',
        'amp-red': '#c41e3a',
        'amp-green': '#2dd4bf',
        'amp-blue': '#3b82f6',
      },
      fontFamily: {
        'display': ['Instrument Sans', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #d4a84b, 0 0 10px #d4a84b' },
          '100%': { boxShadow: '0 0 10px #d4a84b, 0 0 20px #d4a84b, 0 0 30px #d4a84b' },
        },
      },
    },
  },
  plugins: [],
};


