// src/components/ParserForm.js
import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  TextField,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Button,
  Grid,
  Tooltip,
  Typography,
} from '@mui/material';
import { LoadingButton } from '@mui/lab';
import emailTemplates from '../utils/emailTemplates';
import axios from 'axios';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';

function ParserForm() {
  const [template, setTemplate] = useState('');
  const [emailContent, setEmailContent] = useState('');
  const [parserOption, setParserOption] = useState('');
  const [charCount, setCharCount] = useState(0);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const parserOptions = [
    { value: 'hybrid_parser', label: 'Hybrid Parser' },
    { value: 'rule_based', label: 'Rule-Based Parser' },
    { value: 'local_llm', label: 'Local LLM Parser' },
    { value: 'llm', label: "OpenAI LLM Parser" },
  ];

  const handleTemplateChange = (event) => {
    const selectedTemplate = event.target.value;
    setTemplate(selectedTemplate);
    setEmailContent(emailTemplates[selectedTemplate] || '');
    setCharCount(emailTemplates[selectedTemplate]?.length || 0);
  };

  const handleSampleLoad = (templateName) => {
    setTemplate(templateName);
    setEmailContent(emailTemplates[templateName]);
    setCharCount(emailTemplates[templateName].length);
  };

  const handleEmailContentChange = (event) => {
    setEmailContent(event.target.value);
    setCharCount(event.target.value.length);
  };

  const handleParserOptionChange = (event) => {
    setParserOption(event.target.value);
  };

  const validate = () => {
    const newErrors = {};
    if (!emailContent.trim()) newErrors.emailContent = 'Please enter the email content to parse.';
    if (!parserOption) newErrors.parserOption = 'Please select a parser option.';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) return;

    setLoading(true);
    // Implement form submission logic here (e.g., API call)
    // Simulating API call with timeout
    setTimeout(() => {
      setLoading(false);
      // Handle success or error based on response
    }, 2000);
  };

  return (
    <Card>
      <CardHeader title="Input" />
      <CardContent>
        <form onSubmit={handleSubmit} noValidate>
          <Grid container spacing={2}>
            {/* Template Selector */}
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="template-selector-label">Select Template</InputLabel>
                <Select
                  labelId="template-selector-label"
                  id="template_selector"
                  value={template}
                  label="Select Template"
                  onChange={handleTemplateChange}
                >
                  <MenuItem value="">
                    <em>Select a template...</em>
                  </MenuItem>
                  <MenuItem value="meeting">Meeting Invitation</MenuItem>
                  <MenuItem value="invoice">Invoice Email</MenuItem>
                  <MenuItem value="shipping">Shipping Notification</MenuItem>
                  <MenuItem value="claim">Insurance Claim</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {/* Sample Inputs Buttons */}
            <Grid item xs={12}>
              <Typography variant="subtitle1">Or load a sample email:</Typography>
              <Grid container spacing={1} sx={{ mt: 1 }}>
                {['claim', 'meeting', 'invoice', 'shipping'].map((templateName) => (
                  <Grid item key={templateName}>
                    <Tooltip title={`Load ${templateName.charAt(0).toUpperCase() + templateName.slice(1)} Email`}>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => handleSampleLoad(templateName)}
                        aria-label={`Load ${templateName.charAt(0).toUpperCase() + templateName.slice(1)} Email`}
                      >
                        {templateName.charAt(0).toUpperCase() + templateName.slice(1)}
                      </Button>
                    </Tooltip>
                  </Grid>
                ))}
              </Grid>
            </Grid>

            {/* Email Content */}
            <Grid item xs={12}>
              <TextField
                label="Email Content"
                multiline
                rows={10}
                fullWidth
                value={emailContent}
                onChange={handleEmailContentChange}
                error={Boolean(errors.emailContent)}
                helperText={errors.emailContent}
                required
                aria-required="true"
                aria-describedby="char_count"
              />
              <Typography variant="caption" color={charCount > 5000 ? 'error' : 'textSecondary'} id="char_count">
                {charCount} character{charCount !== 1 ? 's' : ''}
              </Typography>
            </Grid>

            {/* Parser Option */}
            <Grid item xs={12}>
              <FormControl fullWidth error={Boolean(errors.parserOption)}>
                <InputLabel id="parser-option-label">Parser Option</InputLabel>
                <Select
                  labelId="parser-option-label"
                  id="parser_option"
                  value={parserOption}
                  label="Parser Option"
                  onChange={handleParserOptionChange}
                  required
                  aria-required="true"
                >
                  <MenuItem value="">
                    <em>Select a parser option...</em>
                  </MenuItem>
                  {parserOptions.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
                {errors.parserOption && (
                  <Typography variant="caption" color="error">
                    {errors.parserOption}
                  </Typography>
                )}
              </FormControl>
            </Grid>

            {/* Submit Button */}
            <Grid item xs={12}>
              <LoadingButton
                type="submit"
                variant="contained"
                color="primary"
                fullWidth
                loading={loading}
                aria-label="Parse Email"
              >
                Parse Email
              </LoadingButton>
            </Grid>
          </Grid>
        </form>
      </CardContent>
    </Card>
  );
}

export default ParserForm;
