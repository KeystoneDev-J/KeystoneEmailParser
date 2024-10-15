// src/components/ParsedData.js
import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Tabs,
  Tab,
  Box,
  Typography,
  Button,
  Alert,
  Snackbar,
} from '@mui/material';
import { ContentCopy, Download, CheckCircle } from '@mui/icons-material';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';
import { CopyToClipboard } from 'react-copy-to-clipboard';

function ParsedData() {
  const [tab, setTab] = useState(0);
  const [parsedData, setParsedData] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMsg, setSnackbarMsg] = useState('');

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
  };

  const handleCopy = () => {
    setSnackbarMsg('Parsed data copied to clipboard!');
    setOpenSnackbar(true);
  };

  // Dummy data for demonstration; replace with actual parsed data
  React.useEffect(() => {
    // Simulate fetching parsed data
    const fetchData = async () => {
      // Replace with actual API call
      const data = {
        subject: 'Team Meeting - Project Update',
        from: 'manager@company.com',
        to: 'team@company.com',
        date: 'March 15, 2024',
        body: 'Hi team, Let\'s meet to discuss the project progress...',
      };
      setParsedData(data);
    };
    fetchData();
  }, []);

  React.useEffect(() => {
    if (parsedData) {
      Prism.highlightAll();
    }
  }, [parsedData, tab]);

  if (!parsedData) {
    return (
      <Card>
        <CardHeader title="Parsed Data" />
        <CardContent>
          <Typography variant="body1">No data parsed yet.</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader title="Parsed Data" />
      <CardContent>
        <Tabs value={tab} onChange={handleTabChange} aria-label="Parsed data tabs">
          <Tab label="JSON" />
          <Tab label="Human-Readable" />
        </Tabs>
        <Box sx={{ mt: 2 }}>
          {tab === 0 && (
            <Box sx={{ position: 'relative' }}>
              <CopyToClipboard text={JSON.stringify(parsedData, null, 2)} onCopy={handleCopy}>
                <Button
                  variant="outlined"
                  startIcon={<ContentCopy />}
                  sx={{ position: 'absolute', top: 0, right: 0 }}
                  aria-label="Copy Parsed Results"
                >
                  Copy
                </Button>
              </CopyToClipboard>
              <pre>
                <code className="language-json">{JSON.stringify(parsedData, null, 2)}</code>
              </pre>
            </Box>
          )}
          {tab === 1 && (
            <Box>
              {/* Render human-readable data */}
              {Object.entries(parsedData).map(([key, value]) => (
                <Typography key={key} variant="body1" gutterBottom>
                  <strong>{key.charAt(0).toUpperCase() + key.slice(1)}:</strong> {value}
                </Typography>
              ))}
            </Box>
          )}
        </Box>
        {/* Success Snackbar */}
        <Snackbar
          open={openSnackbar}
          autoHideDuration={3000}
          onClose={() => setOpenSnackbar(false)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setOpenSnackbar(false)} severity="success" sx={{ width: '100%' }}>
            {snackbarMsg}
          </Alert>
        </Snackbar>
      </CardContent>
    </Card>
  );
}

export default ParsedData;
