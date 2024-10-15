// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { getDesignTokens } from './theme/theme';
import { useMemo, useState, useEffect } from 'react';

function Root() {
  const [mode, setMode] = useState('light');

  // Retrieve theme from localStorage
  useEffect(() => {
    const savedMode = localStorage.getItem('theme') || 'light';
    setMode(savedMode);
  }, []);

  const theme = useMemo(() => createTheme(getDesignTokens(mode)), [mode]);

  const toggleTheme = () => {
    setMode((prevMode) => {
      const nextMode = prevMode === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', nextMode);
      return nextMode;
    });
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App mode={mode} toggleTheme={toggleTheme} />
    </ThemeProvider>
  );
}

ReactDOM.render(<Root />, document.getElementById('root'));
