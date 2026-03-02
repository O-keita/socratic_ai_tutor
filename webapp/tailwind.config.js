/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Primary brand: deep indigo — academic, professional
        brand: {
          50:  '#EEF2FF',
          100: '#E0E7FF',
          200: '#C7D2FE',
          300: '#A5B4FC',
          400: '#818CF8',
          500: '#6366F1',
          600: '#4F46E5',
          700: '#4338CA',
          800: '#3730A3',
          900: '#312E81',
        },
        // Accent: warm orange for CTAs
        accent: {
          50:  '#FFF7ED',
          100: '#FFEDD5',
          200: '#FED7AA',
          400: '#FB923C',
          500: '#F97316',
          600: '#EA580C',
          700: '#C2410C',
        },
        // Page backgrounds
        surface: {
          page: '#F8FAFC',
          card: '#FFFFFF',
          muted: '#F1F5F9',
          border: '#E2E8F0',
        },
        // Navy for topbar/dark elements
        navy: {
          800: '#1E293B',
          900: '#0F172A',
          950: '#020617',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'hero-gradient':    'linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #A855F7 100%)',
        'card-gradient':    'linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%)',
        'accent-gradient':  'linear-gradient(135deg, #F97316 0%, #EA580C 100%)',
        'dark-gradient':    'linear-gradient(180deg, #1E293B 0%, #0F172A 100%)',
      },
      boxShadow: {
        'brand':       '0 4px 24px -4px rgba(79, 70, 229, 0.20)',
        'brand-lg':    '0 8px 40px -6px rgba(79, 70, 229, 0.28)',
        'card':        '0 1px 3px 0 rgba(0,0,0,0.05), 0 1px 2px -1px rgba(0,0,0,0.05)',
        'card-hover':  '0 4px 16px -4px rgba(0,0,0,0.10), 0 2px 6px -2px rgba(0,0,0,0.06)',
        'accent':      '0 4px 20px -4px rgba(249, 115, 22, 0.30)',
        'nav':         '0 1px 0 0 #E2E8F0',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      typography: {
        DEFAULT: {
          css: {
            color: '#334155',
            a: { color: '#4F46E5' },
            h1: { color: '#0F172A' },
            h2: { color: '#0F172A' },
            h3: { color: '#1E293B' },
            h4: { color: '#1E293B' },
            strong: { color: '#0F172A' },
            code: { color: '#4F46E5', background: '#EEF2FF', padding: '2px 6px', borderRadius: '4px' },
            'pre code': { background: 'transparent', padding: '0' },
            pre: { background: '#0F172A', border: '1px solid #E2E8F0' },
            blockquote: { borderLeftColor: '#4F46E5', color: '#64748B' },
            'ul > li::marker': { color: '#4F46E5' },
            'ol > li::marker': { color: '#4F46E5' },
            hr: { borderColor: '#E2E8F0' },
          },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
