// !! We need to REPLACE THIS with the actual URL of your deployed Python/ML Backend API !!
const BACKEND_API_URL = 'http://localhost:5000/api/detect'; 

// --- DOM ELEMENTS ---
const navButtons = document.querySelectorAll('.nav-button');
const pageContents = document.querySelectorAll('.page-content');
const ctaButtons = document.querySelectorAll('.cta-button'); 

const fileInput = document.getElementById('image-upload');
const analyzeButton = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const apiDetailSection = document.getElementById('api-analysis-detail'); 
const loadingSpinner = document.getElementById('loading-spinner');
const resultContent = document.getElementById('result-content');

let uploadedFile = null; 

// --- PAGE NAVIGATION LOGIC ---

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
    
    // Hide the detailed analysis section when switching pages, unless it is the detection page
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

// --- IMAGE ANALYSIS LOGIC (MOCK DATA) ---

fileInput.addEventListener('change', (event) => {
    uploadedFile = event.target.files[0];
    
    if (uploadedFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('image-preview').innerHTML = `<img src="${e.target.result}" alt="Uploaded Leaf Image">`;
        };
        reader.readAsDataURL(uploadedFile);
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
    resultContent.classList.add('hidden'); 
    analyzeButton.disabled = true;

    // --- TEMPORARY MOCK DATA ---
    setTimeout(() => {
        const mockData = {
            disease: "Tomato Late Blight",
            confidence: 0.98,
            remedy: "Immediate application of a systemic fungicide (e.g., Chlorothalonil) and removal of all infected leaves to prevent rapid spore spread. Ensure proper ventilation."
        };
        displayResults(mockData);
        
        loadingSpinner.classList.add('hidden');
        analyzeButton.disabled = false;
        
    }, 2500); 
    // ----------------------------
    
    // ** UNCOMMENT REAL API CALL HERE WHEN READY **
});

// --- Function to update the HTML with the analysis data ---
function displayResults(data) {
    const confidencePercent = (data.confidence * 100).toFixed(2) + '%';
    
    document.getElementById('disease-name').textContent = data.disease || 'Undetermined';
    document.getElementById('confidence').textContent = confidencePercent;
    
    // Update the remedy guide content with the specific result
    const remedyDetail = document.getElementById('remedy-detail');
    remedyDetail.innerHTML = `
        <h2>Remedy for ${data.disease}</h2>
        <p class="placeholder-text">Model Confidence: *${confidencePercent}*</p>
        <div class="guide-card">
            <h4>Recommended Action:</h4>
            <p>${data.remedy}</p>
        </div>
        <button class="cta-button" onclick="showPage('detection')">Back to Analyzer</button>
    `;
    
    // Reveal the detailed analysis section on the Detection page
    apiDetailSection.classList.remove('hidden');

    // Switch to the remedy page after displaying results
    if (data.disease !== 'Analysis Failed') {
        showPage('remedy');
    }
    
    resultContent.classList.remove('hidden');
}