/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#10151c",
        panel: "#171d26",
        panelRaised: "#1d242f",
        borderSubtle: "#2a323d",
        textPrimary: "#e8ecf1",
        textMuted: "#8b96a5",
        accentCyan: "#4fd1c5",
        accentAmber: "#e8b94d",
        accentRose: "#e8607a",
        accentViolet: "#9b8cfb",
      },
      fontFamily: {
        mono: ["IBM Plex Mono", "monospace"],
        sans: ["IBM Plex Sans", "sans-serif"],
      },
    },
  },
  plugins: [],
}

