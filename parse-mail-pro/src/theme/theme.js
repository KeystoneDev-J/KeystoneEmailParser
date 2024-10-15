// src/theme/theme.js
import { createTheme } from '@mui/material/styles';

export const getDesignTokens = (mode) => ({
  palette: {
    mode,
    ...(mode === 'light'
      ? {
          // Light theme palette
          primary: {
            main: '#4f46e5',
          },
          secondary: {
            main: '#6366f1',
          },
          background: {
            default: '#f5f7ff',
            paper: '#ffffff',
          },
          text: {
            primary: '#1f2937',
          },
        }
      : {
          // Dark theme palette
          primary: {
            main: '#6d28d9',
          },
          secondary: {
            main: '#8b5cf6',
          },
          background: {
            default: '#111827',
            paper: '#1f2937',
          },
          text: {
            primary: '#f3f4f6',
          },
        }),
  },
  typography: {
    fontFamily: 'Inter, sans-serif',
  },
  transitions: {
    duration: {
      standard: 300,
    },
  },
});
