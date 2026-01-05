// detector-logic.js

// !!! IMPORTANT: REPLACE this placeholder with your actual Hugging Face Space URL !!!
// This URL must point to your live deployed backend.
const HF_SPACE_BASE_URL = 'https://[YOUR-HF-SPACE-URL-HERE]'; 
const API_PREDICT_ENDPOINT = `${HF_SPACE_BASE_URL}/api/detect-disease`; 
const API_SWITCH_ENDPOINT = `${HF_SPACE_BASE_URL}/api/switch-model`; 
// --------------------------------------------------------------------------

const navButtons = document.querySelectorAll('.nav-button');
const pageContents = document.querySelectorAll('.page-content');
const ctaButtons = document.querySelectorAll('.cta-button'); 

// UI Elements (Ensure these IDs exist in your index.html)
const fileInput = document.getElementById('image-upload');
const analyzeButton = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const apiDetailSection = document.getElementById('api-analysis-detail'); 
const loadingSpinner = document.getElementById('loading-spinner');
const resultContent = document.getElementById('result-content');
const modelSelector = document.getElementById('model-selector'); // New element for A/B testing

let uploadedFile = null; 

// --- 1. PAGE NAVIGATION LOGIC ---

function showPage(pageId) {
    pageContents.forEach(page => page.classList.add('hidden'));

    const targetPage = document.getElementById(`page-${pageId}`);
    if (targetPage) {
        targetPage.classList.remove('hidden');
    }

    navButtons.forEach(button => {
        button.classList.remove('active');
        if (button.getAttribute('data-page') === pageId) {
            button.classList.add('active');
        }
    });
    
    if (pageId !== 'detection') {
        apiDetailSection.classList.add('hidden');
    }
}

navButtons.forEach(button => {
    button.addEventListener('click', () => {
        showPage(button.getAttribute('data-page'));
    });
});

ctaButtons.forEach(button => {
    button.addEventListener('click', () => {
        showPage(button.getAttribute('data-page'));
    });
});

showPage('home'); 

// --- 2. MODEL SWITCHING LOGIC (MLOPS) ---

// Attach listener to the model selector (even if the button is clicked, we use the value)
if (modelSelector) {
    document.querySelector('#model-switch-area button').addEventListener('click', sendSwitchRequest);
}

function sendSwitchRequest() {
    const newModelKey = modelSelector.value;
    
    fetch(API_SWITCH_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: newModelKey })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert(`✅ Server model successfully switched to ${data.active_model}. Future predictions will use it.`);
        } else {
            alert(`❌ Error switching model: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Switching network error:', error);
        alert('Could not connect to the backend API for model switching.');
    });
}


// --- 3. IMAGE UPLOAD & PREDICTION LOGIC ---

fileInput.addEventListener('change', (event) => {
    uploadedFile = event.target.files[0];
    
    if (uploadedFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Updated to use the correct image preview injection logic
            document.getElementById('image-preview').innerHTML = `<img id="result-img" src="${e.target.result}" alt="Uploaded Leaf Image" style="max-width:100%; display:block;">`;
        };
        reader.readAsDataURL(uploadedFile);
        // The analyze button is enabled here, confirming file selection works
        analyzeButton.disabled = false; 
    } else {
        document.getElementById('image-preview').innerHTML = '<p>No image selected yet.</p>';
        analyzeButton.disabled = true; 
    }
    
    resultsSection.classList.add('hidden');
    apiDetailSection.classList.add('hidden'); 
});

analyzeButton.addEventListener('click', async () => {
    if (!uploadedFile) return;

    loadingSpinner.classList.remove('hidden');
    resultsSection.classList.remove('hidden'); 
    analyzeButton.disabled = true;

    // --- API CALL ---
    const formData = new FormData();
    formData.append('file', uploadedFile); // The key 'file' must match app.py

    try {
        const response = await fetch(API_PREDICT_ENDPOINT, {
            method: 'POST', 
            body: formData, 
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = `Server error status: ${response.status}.`;
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.error || errorMessage;
            } catch {
                // If the error response is not JSON, use the status
            }
            throw new Error(errorMessage);
        }

        const data = await response.json(); 
        displayResults(data);

    } catch (error) {
        // Display generic failure if connection or server fails
        displayResults({ disease: 'Connection Failed', model_used: 'N/A', remedy: 'Could not reach server. Check Hugging Face logs.' });
    } finally {
        loadingSpinner.classList.add('hidden');
        analyzeButton.disabled = false;
    }
});


// --- 4. Function to update the HTML with the analysis data ---
function displayResults(data) {
    const confidence = 0.95; // Placeholder since your app.py returns fixed values for now
    const confidencePercent = (data.confidence || confidence * 100).toFixed(2) + '%';
    
    document.getElementById('model-used-display').textContent = data.model_used || 'VGG-19 (Default)';
    document.getElementById('disease-name').textContent = data.disease || 'Undetermined';
    document.getElementById('confidence').textContent = confidencePercent;
    document.getElementById('remedy-text').textContent = data.remedy || 'Check model output.';

    // Populate the Remedy Page dynamically
    const remedyDetail = document.getElementById('remedy-detail');
    remedyDetail.innerHTML = `
        <h2>Remedy for ${data.disease || 'Undetermined Disease'}</h2>
        <p class="placeholder-text">Model used for this prediction: <b>${data.model_used}</b></p>
        <div class="guide-card">
            <h4>Recommended Action:</h4>
            <p>${data.remedy || 'No specific remedy found.'}</p>
        </div>
    `;
    
    // Show the detailed JSON output
    apiDetailSection.classList.remove('hidden');
    apiDetailSection.innerHTML = `<h3>Detailed API Analysis Output</h3><pre>${JSON.stringify(data, null, 2)}</pre>`;
    
    // Switch to the Remedy page if prediction succeeded
    if (data.disease && data.disease !== 'Connection Failed') {
        showPage('remedy');
    }
    
    // Ensure content is visible
    resultsSection.classList.remove('hidden');
    // resultContent.classList.remove('hidden'); // Assuming resultsSection is the main container
}