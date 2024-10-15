// src/components/Navbar.js
import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Tooltip, Button } from '@mui/material';
import { Brightness4, Brightness7, HelpOutline } from '@mui/icons-material';

function Navbar({ mode, toggleTheme }) {
  return (
    <AppBar position="static" color="background.paper" elevation={4}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: 'primary.main' }}>
          Parse Mail Pro
        </Typography>
        <Tooltip title="Toggle light/dark theme">
          <IconButton onClick={toggleTheme} color="inherit" aria-label="Toggle Theme">
            {mode === 'light' ? <Brightness4 /> : <Brightness7 />}
          </IconButton>
        </Tooltip>
        <Button
          variant="outlined"
          startIcon={<HelpOutline />}
          data-bs-toggle="modal"
          data-bs-target="#helpModal"
          aria-label="Open Help Modal"
          sx={{ ml: 2 }}
        >
          Help
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
