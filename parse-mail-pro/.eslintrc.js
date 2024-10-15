// .eslintrc.js
module.exports = {
    env: {
      browser: true,
      es2021: true,
    },
    extends: [
      'eslint:recommended',
      'plugin:react/recommended',
      'prettier',
    ],
    parserOptions: {
      ecmaFeatures: {
        jsx: true,
      },
      ecmaVersion: 12,
      sourceType: 'module',
    },
    plugins: [
      'react',
      'prettier',
    ],
    rules: {
      'prettier/prettier': 'error',
      // Add or modify rules as needed
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
  };
  