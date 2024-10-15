// src/App.js
import React from 'react';
import Navbar from './components/Navbar';
import ParserForm from './components/ParserForm';
import ParsedData from './components/ParsedData';
import HelpModal from './components/HelpModal';
import LoadingOverlay from './components/LoadingOverlay';
import { Container, Grid } from '@mui/material';
import { SnackbarProvider } from 'notistack';

function App({ mode, toggleTheme }) {
  return (
    <SnackbarProvider maxSnack={3}>
      <Navbar mode={mode} toggleTheme={toggleTheme} />
      <Container sx={{ mt: 4, mb: 5 }}>
        <Grid container spacing={4}>
          <Grid item xs={12} lg={6}>
            <ParserForm />
          </Grid>
          <Grid item xs={12} lg={6}>
            <ParsedData />
          </Grid>
        </Grid>
      </Container>
      <HelpModal />
      <LoadingOverlay />
    </SnackbarProvider>
  );
}

export default App;
