// src/components/LoadingOverlay.js
import React from 'react';
import { Backdrop, CircularProgress, Typography, Box } from '@mui/material';
import Lottie from 'lottie-react';
import loadingAnimationData from '../assets/loadingAnimation.json'; // Ensure you have this JSON
import successAnimationData from '../assets/successAnimation.json'; // Ensure you have this JSON

function LoadingOverlay({ open, message }) {
  return (
    <Backdrop open={open} sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Box sx={{ textAlign: 'center' }}>
        <Lottie animationData={loadingAnimationData} loop={true} style={{ width: 150, height: 150 }} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          {message || 'Loading...'}
        </Typography>
      </Box>
    </Backdrop>
  );
}

export default LoadingOverlay;
