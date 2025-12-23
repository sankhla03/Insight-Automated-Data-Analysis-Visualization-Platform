/* ================================
   THEME MANAGEMENT
================================ */

// DEBUGGING: Verify JS file loads
console.log("JS FILE LOADED");

function toggleTheme() {
  const body = document.body;
  const currentTheme = body.getAttribute("data-theme") || "light";
  const newTheme = currentTheme === "light" ? "dark" : "light";

  body.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);
}

function loadTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.body.setAttribute("data-theme", savedTheme);
}

/* ================================
   FILE UPLOAD ENHANCEMENT
================================ */
function setupFileUpload() {
  const fileInput = document.querySelector('input[type="file"]');
  const fileLabel = document.querySelector(".file-upload-label");

  if (!fileInput || !fileLabel) return;

  fileInput.addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file selected";
    fileLabel.textContent = `Selected: ${fileName}`;
    fileLabel.style.borderColor = "var(--accent-color)";
    fileLabel.style.background = "var(--bg-card)";
  });
}

/* ================================
   CHART HOVER EFFECTS
================================ */
function enhanceChartHover() {
  document.addEventListener("mouseover", function (e) {
    const chart = e.target.closest(".js-plotly-plot");
    if (chart) {
      chart.style.transform = "scale(1.02)";
      chart.style.transition = "transform 0.3s ease";
    }
  });

  document.addEventListener("mouseout", function (e) {
    const chart = e.target.closest(".js-plotly-plot");
    if (chart) {
      chart.style.transform = "scale(1)";
    }
  });
}

/* ================================
   LOADING STATE
================================ */
function showLoading(button) {
  const originalText = button.textContent;
  button.innerHTML = "⏳ Loading...";
  button.disabled = true;

  return function hideLoading() {
    button.textContent = originalText;
    button.disabled = false;
  };
}

/* ================================
   FORM VALIDATION
================================ */
function validateForm(formId) {
  const form = document.getElementById(formId);
  if (!form) return true;

  let isValid = true;
  form.querySelectorAll("[required]").forEach((field) => {
    if (!field.value.trim()) {
      field.style.borderColor = "var(--error)";
      isValid = false;
    } else {
      field.style.borderColor = "var(--border-color)";
    }
  });

  return isValid;
}

/* ================================
   COLUMN OPTIONS HELPER
================================ */
function updateColumnOptions(selectElement, columns) {
  selectElement.innerHTML = '<option value="">Select column</option>';
  columns.forEach((col) => {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    selectElement.appendChild(option);
  });
}

/* ================================
   CARD ANIMATIONS
================================ */
function animateCards() {
  document.querySelectorAll(".card").forEach((card, index) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(20px)";

    setTimeout(() => {
      card.style.transition = "all 0.5s ease";
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    }, index * 100);
  });
}

/* ================================
   FEEDBACK SYSTEM
================================ */
function submitFeedback(feedbackText) {
  const textarea = document.querySelector(
    "#feedback-form textarea[name='feedback_text']"
  );
  if (textarea) {
    textarea.value = feedbackText;
    showConfirmationMessage("Feedback submitted successfully!");
  }
}

function showConfirmationMessage(message) {
  const msg = document.createElement("div");
  msg.className = "message success";
  msg.textContent = message;
  msg.style.cssText =
    "position:fixed;top:20px;right:20px;z-index:9999";

  document.body.appendChild(msg);
  setTimeout(() => msg.remove(), 3000);
}

/* ================================
   DATA TABLE ENHANCEMENT
================================ */
function enhanceDataTables() {
  document.querySelectorAll("table th").forEach((th) => {
    th.style.cursor = "pointer";
    th.addEventListener("click", () => {
      console.log("Sorting column:", th.textContent);
    });
  });
}

/* ================================
   🔥 VISUALIZATION LOGIC (FIXED)
================================ */
function setupVisualizationControls() {
  // DEBUGGING: Log initialization
  console.log("Visualization controls initialized");
  
  const chartType = document.getElementById("chart-type-select");
  const xSelect = document.getElementById("x-column-select");
  const ySelect = document.getElementById("y-column-select");
  const colorSelect = document.getElementById("color-column-select");
  // const xLabel = document.querySelector('label[for="x-column-select"]');
  // const yLabel = document.querySelector('label[for="y-column-select"]');
  // const colorLabel = document.getElementById("color-label");

  if (!chartType || !xSelect || !ySelect) return;

  // Chart type configurations
  const chartConfigs = {
    histogram: {
      x: { enabled: true, required: true, label: "X-axis Column:" },
      y: { enabled: false, required: false, label: "Y-axis Column:" },
      color: { enabled: true, required: false, label: "Color Column:" }
    },
    box: {
      x: { enabled: true, required: false, label: "X-axis Column:" },
      y: { enabled: true, required: true, label: "Y-axis Column:" },
      color: { enabled: true, required: false, label: "Color Column:" }
    },
    pie: {
      x: { enabled: true, required: true, label: "Category Column:" },
      y: { enabled: false, required: false, label: "Y-axis Column:" },
      color: { enabled: true, required: false, label: "Color Column:" }
    },
    line: {
      x: { enabled: true, required: true, label: "X-axis Column:" },
      y: { enabled: true, required: true, label: "Y-axis Column:" },
      color: { enabled: true, required: false, label: "Color Column:" }
    },
    scatter: {
      x: { enabled: true, required: true, label: "X-axis Column:" },
      y: { enabled: true, required: true, label: "Y-axis Column:" },
      color: { enabled: true, required: false, label: "Color Column:" }
    },
    map: {
      x: { enabled: false, required: false, label: "X-axis Column:" },
      y: { enabled: false, required: false, label: "Y-axis Column:" },
      color: { enabled: true, required: true, label: "Location Column:" }
    }
  };

  function resetAllFields() {
    // Reset all fields to default state
    [xSelect, ySelect, colorSelect].forEach((el) => {
      el.disabled = false;
      el.required = false;
      el.parentElement.style.display = "block";
      el.value = ""; // Clear values
    });
  }

  function updateFieldState(field, config) {
    if (!field) return;

    if (config.enabled) {
      field.disabled = false;
      field.parentElement.style.display = "block";
      field.required = config.required;
    } else {
      field.disabled = true;
      field.required = false;
      field.value = ""; // Clear value when disabled
    }
  }

  function updateFieldLabel(field, label, newLabel) {
    // Removed label updating to prevent null reference errors
    // Field labels are static in the HTML and don't need dynamic updating
  }

  function updateChartUI(chartType) {
    resetAllFields();

    if (!chartType || !chartConfigs[chartType]) {
      return; // Invalid chart type, keep default state
    }

    const config = chartConfigs[chartType];

    // Update X field
    updateFieldState(xSelect, config.x);
    // Removed label update to prevent null reference errors

    // Update Y field
    updateFieldState(ySelect, config.y);
    // Removed label update to prevent null reference errors

    // Update Color field
    updateFieldState(colorSelect, config.color);
    // Removed label update to prevent null reference errors

    // Add visual feedback for required fields
    [xSelect, ySelect, colorSelect].forEach((field) => {
      const label = field.parentElement.querySelector('label');
      if (label) {
        label.style.fontWeight = field.required ? 'bold' : 'normal';
        label.style.color = field.required ? 'var(--accent-color)' : 'inherit';
      }
    });
  }

  function createVisualization() {
    // DEBUGGING: Log function call
    console.log("Creating visualization", {
      chart: chartType.value,
      x: xSelect.value,
      y: ySelect.value,
      color: colorSelect.value
    });
    
    const selectedChartType = chartType.value;
    if (!selectedChartType) return;
    
    // Guard conditions to prevent empty AJAX calls
    if (!xSelect.value && selectedChartType !== "map") return;

    // Get current field values
    const xColumn = xSelect.value;
    const yColumn = ySelect.value;
    const colorColumn = colorSelect.value;

    // Validate required fields based on chart type
    const config = chartConfigs[selectedChartType];
    let isValid = true;
    let missingFields = [];

    if (config.x.required && !xColumn) {
      isValid = false;
      missingFields.push("X-axis Column");
    }
    if (config.y.required && !yColumn) {
      isValid = false;
      missingFields.push("Y-axis Column");
    }
    if (config.color.required && !colorColumn) {
      isValid = false;
      missingFields.push("Color/Location Column");
    }

    if (!isValid) {
      showValidationError(`Please select: ${missingFields.join(", ")}`);
      return;
    }

    // Show loading state
    showVisualizationLoading(true);

    // Create form data for AJAX request
    const formData = new FormData();
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
    
    if (csrfToken) {
      formData.append('csrfmiddlewaretoken', csrfToken);
    }
    formData.append('chart_type', selectedChartType);
    
    if (xColumn) formData.append('x_column', xColumn);
    if (yColumn) formData.append('y_column', yColumn);
    if (colorColumn) formData.append('color_column', colorColumn);

    // Make AJAX request
    fetch('/ajax/visualization/', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      showVisualizationLoading(false);
      
      if (data.success && data.chart_div && data.chart_script) {
        displayChart(data.chart_div, data.chart_script);
        showVisualizationMessage(data.message || 'Chart created successfully!', 'success');
      } else {
        showVisualizationMessage(data.error || 'Failed to create visualization', 'error');
      }
    })
    .catch(error => {
      showVisualizationLoading(false);
      console.error('Visualization error:', error);
      showVisualizationMessage('Error creating visualization: ' + error.message, 'error');
    });
  }

  function showVisualizationLoading(show) {
    const loading = document.getElementById('chart-loading');
    if (loading) {
      loading.style.display = show ? 'block' : 'none';
    }
  }

  function showValidationError(message) {
    showVisualizationMessage(message, 'error');
  }

  function showVisualizationMessage(message, type) {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.chart-message');
    existingMessages.forEach(msg => msg.remove());

    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `chart-message ${type === 'success' ? 'chart-success' : 'chart-error'}`;
    messageDiv.style.cssText = 'padding: 15px; margin: 10px 0; border-radius: 6px; font-weight: bold;';
    messageDiv.textContent = message;

    // Insert after the visualization form
    const form = document.getElementById('visualization-form-ajax');
    if (form) {
      form.parentNode.insertBefore(messageDiv, form.nextSibling);
    }

    // Auto-hide after 5 seconds
    setTimeout(() => {
      if (messageDiv.parentNode) {
        messageDiv.remove();
      }
    }, 5000);
  }

  function displayChart(chartDiv, chartScript, chartCSS = '') {
    console.log('Displaying chart:', { chartDiv: !!chartDiv, chartScript: !!chartScript }); // Debug log
    
    // Remove any existing chart containers to prevent conflicts
    const existingContainers = document.querySelectorAll('#chart-container, .chart-container');
    existingContainers.forEach(container => container.remove());

    // Create a new chart container
    const container = document.createElement('div');
    container.id = 'chart-container';
    container.style.cssText = 'min-height: 400px; border: 1px solid var(--border-color); border-radius: 6px; margin: 20px 0; padding: 10px;';

    // Insert after the visualization form
    const form = document.getElementById('visualization-form-ajax');
    if (form) {
      form.parentNode.insertBefore(container, form.nextSibling);
    } else {
      // Fallback: append to main content area
      const main = document.querySelector('.container');
      if (main) {
        main.appendChild(container);
      }
    }

    // Add CSS if provided
    if (chartCSS) {
      const styleElement = document.createElement('style');
      styleElement.textContent = chartCSS;
      document.head.appendChild(styleElement);
    }

    // Create the chart HTML
    const chartHTML = chartDiv + (chartScript || '');
    container.innerHTML = chartHTML;

    // Force chart rendering with delay to ensure DOM is ready
    setTimeout(() => {
      try {
        // Find any Plotly graphs in the container and resize them
        const plotlyDivs = container.querySelectorAll('.js-plotly-plot');
        plotlyDivs.forEach(div => {
          if (typeof Plotly !== 'undefined' && Plotly.Plots) {
            console.log('Resizing Plotly chart:', div.id); // Debug log
            Plotly.Plots.resize(div);
          }
        });
        
        // Trigger window resize to ensure proper rendering
        window.dispatchEvent(new Event('resize'));
        
        console.log('Chart rendering completed'); // Debug log
      } catch (error) {
        console.error('Error during chart rendering:', error); // Debug log
      }
    }, 200);

    console.log('Chart display function completed'); // Debug log
  }

  // Event Listeners
  chartType.addEventListener("change", function () {
    updateChartUI(this.value);
    
    // Auto-create chart if we have the minimum required fields
    const selectedChartType = this.value;
    if (selectedChartType && chartConfigs[selectedChartType]) {
      const config = chartConfigs[selectedChartType];
      
      // Check if we have required fields
      const hasX = config.x.required ? xSelect.value : true;
      const hasY = config.y.required ? ySelect.value : true;
      const hasColor = config.color.required ? colorSelect.value : true;
      
      if (hasX && hasY && hasColor) {
        createVisualization();
      }
    }
  });

  // Trigger visualization when fields change
  [xSelect, ySelect, colorSelect].forEach(field => {
    field.addEventListener('change', createVisualization);
    field.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        createVisualization();
      }
    });
  });

  // Initialize with current chart type
  if (chartType.value) {
    updateChartUI(chartType.value);
  }
}

/* ================================
   DOM READY
================================ */
document.addEventListener("DOMContentLoaded", function () {
  loadTheme();
  setupFileUpload();
  enhanceChartHover();
  animateCards();
  enhanceDataTables();
  setupVisualizationControls();

  document
    .querySelector(".theme-toggle")
    ?.addEventListener("click", toggleTheme);
});

/* ================================
   EXPORT GLOBALS
================================ */
window.toggleTheme = toggleTheme;
window.showLoading = showLoading;
window.validateForm = validateForm;
window.updateColumnOptions = updateColumnOptions;
window.submitFeedback = submitFeedback;
window.showConfirmationMessage = showConfirmationMessage;
