// static\js\scripts.js

/* =========================================
   CONSTANTS & VARIABLES
========================================= */

// Loading Messages Array
const LOADING_MESSAGES = [
  "Hold onto your bits, this might take a hot second... 🔥",
  "Teaching AI to read emails... it's like training a cat to swim 🐱",
  "Holy sh*t, this LLM is taking its sweet time... 🐌",
  "Parsing emails faster than your ex replies to texts... which isn't saying much 📱",
  "Making the hamsters run faster in our quantum computers... 🐹",
  "If this takes any longer, we might need to sacrifice a keyboard to the tech gods ⌨️",
  "Currently bribing the AI with virtual cookies 🍪",
  "Plot twist: The AI is actually your old Nokia trying its best 📱",
  "Damn, this is taking longer than explaining NFTs to your grandma 👵",
  "Our AI is having an existential crisis... again 🤖",
  "Loading... like your patience, probably 😅",
  "Working harder than a cat trying to bury poop on a marble floor 🐱",
  "Processing faster than your dating app matches ghost you 👻",
  "Better grab a coffee, this sh*t's taking its time ☕",
];

// Email Templates
const EMAIL_TEMPLATES = {
  meeting: `Subject: Team Meeting - Project Update
From: manager@company.com
To: team@company.com
Date: March 15, 2024

Hi team,

Let's meet to discuss the project progress. The meeting is scheduled for March 20, 2024 at 2:00 PM EST in Conference Room A.

Agenda:
1. Project timeline review
2. Resource allocation
3. Next steps

Please confirm your attendance.

Best regards,
Manager`,
  invoice: `Subject: Invoice #INV-2024-001
From: billing@supplier.com
To: accounts@company.com
Date: March 16, 2024

Dear Customer,

Please find attached invoice #INV-2024-001 for recent services.

Amount Due: $1,500.00
Due Date: March 30, 2024

Payment Details:
Bank: FirstBank
Account: 1234567890
Reference: INV-2024-001

Thank you for your business!

Regards,
Billing Team`,
  shipping: `Subject: Your Order Has Shipped!
From: orders@store.com
To: customer@email.com
Date: March 17, 2024

Dear Customer,

Your order #ORD123456 has shipped!

Tracking Number: 1Z999AA1234567890
Carrier: UPS
Estimated Delivery: March 20, 2024

Order Details:
- Product A ($99.99)
- Product B ($149.99)

Track your package here: https://tracking.ups.com

Thank you for shopping with us!

Best regards,
Store Team`,
  claim: `Subject: Insurance Claim Submission
From: john.doe@example.com
To: claims@insurancecompany.com
Date: April 10, 2024

Dear Claims Handler,

I am writing to submit a claim regarding the recent loss at my property.

**Requesting Party**
Insurance Company: ABC Insurance
Handler: Jane Smith
Carrier Claim Number: CLM-2024-7890

**Insured Information**
Name: John Doe
Contact #: (555) 123-4567
Loss Address: 123 Elm Street, Springfield, IL
Public Adjuster: Mark Johnson
Is the insured an Owner or a Tenant of the loss location? Owner

**Adjuster Information**
Adjuster Name: Emily Davis
Adjuster Phone Number: (555) 987-6543
Adjuster Email: emily.davis@insurancecompany.com
Job Title: Senior Adjuster
Address: 456 Oak Avenue, Springfield, IL
Policy #: POL-567890

**Assignment Information**
Date of Loss/Occurrence: April 5, 2024
Cause of loss: Hail Storm
Facts of Loss: On April 5th, a severe hail storm caused significant damage to the roof and siding of my home.
Loss Description: Multiple shingles were torn off, and several sections of the siding are cracked or missing.
Residence Occupied During Loss: Yes
Was Someone home at time of damage: Yes
Repair or Mitigation Progress: Initial assessments completed; awaiting approval for repairs.
Type: Hail
Inspection type: On-site inspection

Check the box of applicable assignment type:
- [x] Wind
- [x] Structural
- [x] Hail
- [ ] Foundation
- [ ] Other - provide details:

**Additional details/Special Instructions:**
Please expedite the inspection process as repairs are urgently needed to prevent further damage.

**Attachment(s):**
- Photos of damage
- Initial repair estimates

Thank you for your prompt attention to this matter.

Sincerely,
John Doe`,
};

/* =========================================
   STATE MANAGEMENT
========================================= */

// Theme Variables
let currentTheme = "light";

// Loading Variables
let currentMessageIndex = 0;
let loadingInterval;
let progressValue = 0;

// Global array to store parsed entries
const parsedEntries = [];

// Animations
let loadingAnimation;
let successAnimation;

/* =========================================
   DOM CACHING
========================================= */

// Cache frequently accessed DOM elements
const domCache = {
  lottieContainer: document.getElementById("lottie-container"),
  successAnimationContainer: document.getElementById("success-animation"),
  parserForm: document.getElementById("parserForm"),
  templateSelector: document.getElementById("template_selector"),
  sampleButtons: document.querySelectorAll("[data-template]"),
  emailContent: document.getElementById("email_content"),
  charCount: document.getElementById("char_count"),
  themeToggle: document.getElementById("theme-toggle-btn"),
  themeIcon: document.getElementById("theme-icon"),
  downloadCsvBtn: document.getElementById("downloadCsvBtn"),
  downloadPdfBtn: document.getElementById("downloadPdfBtn"),
  parserOption: document.getElementById("parser_option"),
  copyResultsBtn: document.getElementById("copyResultsBtn"),
  loadingOverlay: document.querySelector(".loading-overlay"),
  loadingMessage: document.getElementById("loading-message"),
  progressBar: document.getElementById("progress-bar"),
  jsonOutput: document.getElementById("jsonOutput"),
  humanOutput: document.getElementById("humanOutput"),
  successMessage: document.getElementById("successMessage"),
  errorMessage: document.getElementById("errorMessage"),
};

/* =========================================
   INITIALIZATION
========================================= */

// Initialize on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  initializeAnimations();
  initializeEventListeners();
  initializeTheme();
  initializeTooltips();
});

/* =========================================
   FUNCTIONS
========================================= */

/**
 * Initializes Lottie Animations
 */
function initializeAnimations() {
  if (domCache.lottieContainer) {
    loadingAnimation = lottie.loadAnimation({
      container: domCache.lottieContainer,
      renderer: "svg",
      loop: true,
      autoplay: false,
      path: "https://lottie.host/0c1a139c-8469-489f-a94e-d6f8e379b066/8eOki65eVz.json",
    });
  }

  if (domCache.successAnimationContainer) {
    successAnimation = lottie.loadAnimation({
      container: domCache.successAnimationContainer,
      renderer: "svg",
      loop: false,
      autoplay: false,
      path: "https://assets3.lottiefiles.com/packages/lf20_jbrw3hcz.json",
    });
  }
}

/**
 * Initializes Event Listeners
 */
function initializeEventListeners() {
  if (domCache.parserForm) {
    domCache.parserForm.addEventListener("submit", handleFormSubmission);
  }

  if (domCache.templateSelector) {
    domCache.templateSelector.addEventListener("change", loadTemplate);
  }

  if (domCache.sampleButtons) {
    domCache.sampleButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const templateName = button.getAttribute("data-template");
        loadSampleEmail(templateName);
      });
    });
  }

  if (domCache.emailContent) {
    domCache.emailContent.addEventListener("input", debounce(updateCharCount, 300));
  }

  if (domCache.themeToggle) {
    domCache.themeToggle.addEventListener("click", toggleTheme);
  }

  if (domCache.downloadCsvBtn) {
    domCache.downloadCsvBtn.addEventListener("click", downloadCSV);
  }

  if (domCache.downloadPdfBtn) {
    domCache.downloadPdfBtn.addEventListener("click", downloadPDF);
  }

  if (domCache.parserOption) {
    domCache.parserOption.addEventListener("change", validateFormInputs);
  }

  if (domCache.copyResultsBtn) {
    domCache.copyResultsBtn.addEventListener("click", copyResults);
  }
}

/**
 * Initializes Theme based on localStorage or default
 */
function initializeTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  currentTheme = savedTheme;
  updateThemeIcon();
}

/**
 * Initializes Bootstrap Tooltips
 */
function initializeTooltips() {
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
  tooltipTriggerList.forEach((tooltipTriggerEl) => {
    new bootstrap.Tooltip(tooltipTriggerEl);
  });
}

/**
 * Toggles between light and dark themes
 */
function toggleTheme() {
  currentTheme = currentTheme === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("theme", currentTheme);
  updateThemeIcon();
}

/**
 * Updates the theme toggle icon based on current theme
 */
function updateThemeIcon() {
  if (domCache.themeIcon) {
    domCache.themeIcon.textContent = currentTheme === "light" ? "🌙" : "☀️";
  }
}

/**
 * Loads the selected email template into the textarea
 */
function loadTemplate() {
  if (domCache.templateSelector && domCache.emailContent) {
    domCache.emailContent.value = EMAIL_TEMPLATES[domCache.templateSelector.value] || "";
    updateCharCount();
  }
}

/**
 * Loads a sample email based on the provided template name
 * @param {string} templateName 
 */
function loadSampleEmail(templateName) {
  if (EMAIL_TEMPLATES[templateName]) {
    domCache.emailContent.value = EMAIL_TEMPLATES[templateName];
    updateCharCount();
  }
}

/**
 * Updates the character count display
 */
function updateCharCount() {
  if (domCache.emailContent && domCache.charCount) {
    const count = domCache.emailContent.value.length;
    domCache.charCount.textContent = `${count} character${count !== 1 ? 's' : ''}`;
    domCache.charCount.className = count > 5000 ? "text-danger" : "text-muted";
  }
}

/**
 * Displays the loading overlay with animations
 */
function showLoadingOverlay() {
  if (domCache.loadingOverlay && domCache.loadingMessage && domCache.progressBar) {
    domCache.loadingOverlay.classList.remove("d-none");
    if (loadingAnimation) loadingAnimation.play();

    progressValue = 0;
    domCache.progressBar.style.width = "0%";

    updateLoadingMessage();

    loadingInterval = setInterval(() => {
      currentMessageIndex = (currentMessageIndex + 1) % LOADING_MESSAGES.length;
      updateLoadingMessage();

      progressValue = Math.min(progressValue + 5, 95);
      domCache.progressBar.style.width = `${progressValue}%`;
    }, 2000);
  }
}

/**
 * Hides the loading overlay and stops animations
 */
function hideLoadingOverlay() {
  if (domCache.loadingOverlay && domCache.progressBar) {
    domCache.progressBar.style.width = "100%";

    setTimeout(() => {
      domCache.loadingOverlay.classList.add("d-none");
      if (loadingAnimation) loadingAnimation.stop();
      clearInterval(loadingInterval);
      currentMessageIndex = 0;
    }, 700);
  }
}

/**
 * Updates the loading message with a fade effect
 */
function updateLoadingMessage() {
  if (domCache.loadingMessage) {
    domCache.loadingMessage.classList.remove("visible");

    setTimeout(() => {
      domCache.loadingMessage.textContent = LOADING_MESSAGES[currentMessageIndex];
      domCache.loadingMessage.classList.add("visible");
    }, 300);
  }
}

/**
 * Copies the JSON parsed results to the clipboard
 */
function copyResults() {
  if (domCache.jsonOutput) {
    navigator.clipboard
      .writeText(domCache.jsonOutput.textContent)
      .then(() => {
        showSuccessMessage("Parsed data copied to clipboard!");
        playSuccessAnimation();
      })
      .catch(() => {
        showErrorMessage("Failed to copy to clipboard.");
      });
  }
}

/**
 * Shows a success message to the user
 * @param {string} message 
 */
function showSuccessMessage(message) {
  if (domCache.successMessage) {
    domCache.successMessage.textContent = message;
    domCache.successMessage.classList.remove("d-none");
    setTimeout(() => {
      domCache.successMessage.classList.add("d-none");
    }, 5000);
  }
}

/**
 * Shows an error message to the user
 * @param {string} message 
 */
function showErrorMessage(message) {
  if (domCache.errorMessage) {
    domCache.errorMessage.textContent = message;
    domCache.errorMessage.classList.remove("d-none");
    setTimeout(() => {
      domCache.errorMessage.classList.add("d-none");
    }, 5000);
  }
}

/**
 * Plays the success animation
 */
function playSuccessAnimation() {
  if (domCache.successAnimationContainer && successAnimation) {
    domCache.successAnimationContainer.classList.remove("d-none");
    successAnimation.goToAndPlay(0);
    successAnimation.addEventListener("complete", () => {
      domCache.successAnimationContainer.classList.add("d-none");
    }, { once: true });
  }
}

/**
 * Downloads the parsed data as a CSV file
 */
function downloadCSV() {
  if (parsedEntries.length === 0) {
    showErrorMessage("No parsed data available to download.");
    return;
  }

  const headers = Object.keys(parsedEntries[parsedEntries.length - 1]);
  const csvContent = [
    headers.join(","),
    ...parsedEntries.map(entry =>
      headers.map(header => {
        let cell = entry[header];
        if (typeof cell === "object" && cell !== null) {
          cell = JSON.stringify(cell);
        }
        cell = String(cell).replace(/"/g, '""');
        return /[",\n]/.test(cell) ? `"${cell}"` : cell;
      }).join(",")
    ),
  ].join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  const timestamp = new Date().toISOString().replace(/[:\-T.]/g, "").split("Z")[0];
  link.setAttribute("download", `parsed_emails_${timestamp}.csv`);
  link.style.visibility = "hidden";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showSuccessMessage("CSV downloaded successfully!");
  playSuccessAnimation();
}

/**
 * Downloads the parsed data as a PDF file
 */
async function downloadPDF() {
  if (parsedEntries.length === 0) {
    showErrorMessage("No parsed data available to download.");
    return;
  }

  if (!window.jspdf || !window.jspdf.jsPDF) {
    showErrorMessage("PDF library not loaded.");
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  // Add title
  doc.setFontSize(16);
  doc.text("Parsed Email Data", 10, 10);

  // Add JSON data
  doc.setFontSize(12);
  const jsonString = JSON.stringify(parsedEntries[parsedEntries.length - 1], null, 2);
  const lines = doc.splitTextToSize(jsonString, 180);
  doc.text(lines, 10, 20);

  // Save the PDF
  doc.save(`parsed_emails_${new Date().toISOString().split("T")[0]}.pdf`);

  showSuccessMessage("PDF downloaded successfully!");
  playSuccessAnimation();
}

/**
 * Displays the parsed data in both JSON and Human-Readable formats
 * @param {Object} parsedData 
 */
function displayParsedData(parsedData) {
  if (domCache.jsonOutput) {
    const prettyJson = JSON.stringify(parsedData, null, 2);
    domCache.jsonOutput.textContent = prettyJson;
    Prism.highlightElement(domCache.jsonOutput);
  }

  if (domCache.humanOutput) {
    domCache.humanOutput.innerHTML = renderHumanReadable(parsedData);
  }

  parsedEntries.push(flattenParsedData(parsedData));

  if (domCache.downloadCsvBtn && domCache.downloadPdfBtn) {
    domCache.downloadCsvBtn.classList.remove("d-none");
    domCache.downloadPdfBtn.classList.remove("d-none");
  }
}

/**
 * Flattens nested parsed data for CSV export
 * @param {Object} data 
 * @returns {Object}
 */
function flattenParsedData(data) {
  const flatData = {};
  for (const [section, content] of Object.entries(data)) {
    if (typeof content === "object" && content !== null) {
      for (const [key, value] of Object.entries(content)) {
        flatData[`${capitalizeFirstLetter(section)} - ${capitalizeFirstLetter(key)}`] = value;
      }
    } else {
      flatData[capitalizeFirstLetter(section)] = content;
    }
  }
  return flatData;
}

/**
 * Renders the parsed data in a human-readable accordion format
 * @param {Object} data 
 * @returns {string}
 */
function renderHumanReadable(data) {
  let htmlContent = "<div class='human-readable-container'><div class='accordion' id='parsedDataAccordion'>";
  let sectionIndex = 0;
  for (const [section, content] of Object.entries(data)) {
    const collapseId = `collapseSection${sectionIndex}`;
    htmlContent += `
      <div class="accordion-item">
        <h2 class="accordion-header" id="heading${sectionIndex}">
          <button class="accordion-button ${sectionIndex !== 0 ? 'collapsed' : ''}" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="${sectionIndex === 0 ? 'true' : 'false'}" aria-controls="${collapseId}">
            ${capitalizeFirstLetter(section)}
          </button>
        </h2>
        <div id="${collapseId}" class="accordion-collapse collapse ${sectionIndex === 0 ? 'show' : ''}" aria-labelledby="heading${sectionIndex}" data-bs-parent="#parsedDataAccordion">
          <div class="accordion-body">
    `;
    if (typeof content === "object" && content !== null) {
      htmlContent += "<ul>";
      for (const [key, value] of Object.entries(content)) {
        htmlContent += `<li><strong>${capitalizeFirstLetter(key)}:</strong> ${escapeHtml(value)}</li>`;
      }
      htmlContent += "</ul>";
    } else {
      htmlContent += `<p>${escapeHtml(content)}</p>`;
    }
    htmlContent += `
          </div>
        </div>
      </div>
    `;
    sectionIndex++;
  }
  htmlContent += "</div></div>";
  return htmlContent;
}

/**
 * Capitalizes the first letter of a string
 * @param {string} string 
 * @returns {string}
 */
function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

/**
 * Escapes HTML characters to prevent XSS
 * @param {string} text 
 * @returns {string}
 */
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
}

/**
 * Handles the form submission for parsing emails
 * @param {Event} e 
 */
async function handleFormSubmission(e) {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const parserOption = domCache.parserOption.value;
  const emailContent = domCache.emailContent.value.trim();

  // Validate Email Content
  if (!emailContent) {
    toggleInvalidState(domCache.emailContent, true);
    showErrorMessage("Please enter the email content to parse.");
    return;
  } else {
    toggleInvalidState(domCache.emailContent, false);
  }

  // Validate Parser Option
  if (!parserOption) {
    toggleInvalidState(domCache.parserOption, true);
    showErrorMessage("Please select a parser option.");
    return;
  } else {
    toggleInvalidState(domCache.parserOption, false);
  }

  showLoadingOverlay();

  try {
    const response = await fetch("/parse_email", {
      method: "POST",
      body: formData,
    });

    hideLoadingOverlay();

    const contentType = response.headers.get("Content-Type");
    if (contentType && contentType.includes("application/json")) {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error_message || "An error occurred while parsing.");
      }
      displayParsedData(data);
      showSuccessMessage("Email parsed successfully!");
      playSuccessAnimation();
    } else {
      throw new Error("Unexpected response format.");
    }
  } catch (error) {
    console.error("Error during parsing:", error);
    hideLoadingOverlay();
    showErrorMessage(error.message);
    hideDownloadButtons();
  }
}

/**
 * Toggles the invalid state of a form input
 * @param {HTMLElement} element 
 * @param {boolean} isInvalid 
 */
function toggleInvalidState(element, isInvalid) {
  if (element) {
    if (isInvalid) {
      element.classList.add("is-invalid");
    } else {
      element.classList.remove("is-invalid");
    }
  }
}

/**
 * Validates form inputs on change
 */
function validateFormInputs() {
  const emailContent = domCache.emailContent;
  const parserOption = domCache.parserOption;

  // Validate Email Content
  if (emailContent.value.trim() === "") {
    emailContent.classList.add("is-invalid");
  } else {
    emailContent.classList.remove("is-invalid");
  }

  // Validate Parser Option
  if (parserOption.value.trim() === "") {
    parserOption.classList.add("is-invalid");
  } else {
    parserOption.classList.remove("is-invalid");
  }
}

/**
 * Hides the download buttons
 */
function hideDownloadButtons() {
  if (domCache.downloadCsvBtn) {
    domCache.downloadCsvBtn.classList.add("d-none");
  }
  if (domCache.downloadPdfBtn) {
    domCache.downloadPdfBtn.classList.add("d-none");
  }
}

/**
 * Debounce function to limit how often a function can fire.
 * @param {Function} func 
 * @param {number} wait 
 * @returns {Function}
 */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    const later = () => {
      clearTimeout(timeout);
      func.apply(this, args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
