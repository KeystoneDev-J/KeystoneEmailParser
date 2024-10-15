// src/components/HelpModal.js
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';

function HelpModal() {
  const [open, setOpen] = React.useState(false);

  // Listen to Bootstrap's modal trigger via data attributes
  React.useEffect(() => {
    const handleOpen = () => setOpen(true);
    const handleClose = () => setOpen(false);
    const helpButton = document.querySelector('[data-bs-target="#helpModal"]');

    if (helpButton) {
      helpButton.addEventListener('click', handleOpen);
    }

    return () => {
      if (helpButton) {
        helpButton.removeEventListener('click', handleClose);
      }
    };
  }, []);

  return (
    <Dialog open={open} onClose={() => setOpen(false)} aria-labelledby="help-dialog-title" maxWidth="lg" fullWidth>
      <DialogTitle id="help-dialog-title">How to Use Parse Mail Pro</DialogTitle>
      <DialogContent dividers>
        <Typography variant="body1" gutterBottom>
          Welcome to Parse Mail Pro! Here's how to get started:
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Select a Template"
              secondary="Choose an email template from the dropdown to load predefined content."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Load a Sample Email"
              secondary="Alternatively, load a sample claim-related email to see how parsing works."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Input Email Content"
              secondary="If you prefer, you can manually enter or paste your email content into the textarea."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Select Parser Option"
              secondary="Choose the parsing method that best suits your needs."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Parse Email"
              secondary="Click the 'Parse Email' button to process your input."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="View Results"
              secondary="The parsed data will appear in the right section. You can switch between JSON and a human-readable format, copy the results, or download them as CSV/PDF."
            />
          </ListItem>
        </List>
        <Typography variant="body1">
          If you encounter any issues, please refer to the FAQs or contact support.
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setOpen(false)} variant="contained" color="primary" aria-label="Close Help Modal">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default HelpModal;
