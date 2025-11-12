// THIS IS THE CORRECT ENDPOINT ADDRESS - MUST MATCH app.py
const BACKEND_API_URL = 'https://expert-meme-69v4xgjw7wpqh476j-5000.app.github.dev/api/detect';

const navButtons = document.querySelectorAll('.nav-button');
const pageContents = document.querySelectorAll('.page-content');
const ctaButtons = document.querySelectorAll('.cta-button'); 

const fileInput = document.getElementById('image-upload');
const analyzeButton = document.getElementById('analyze-btn');
const imagePreview = document.getElementById('image-preview');
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

// --- IMAGE ANALYSIS LOGIC ---

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

    // --- REAL API CALL ---
    const formData = new FormData();
    formData.append('file', uploadedFile); 

    try {
        // This fetch call uses the correct POST method and URL
        const response = await fetch(BACKEND_API_URL, {
            method: 'POST', 
            body: formData, 
        });

        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }

        const data = await response.json(); 
        displayResults(data);

    } catch (error) {
        displayResults({ disease: 'Connection Failed', confidence: 0, remedy: 'Could not reach server. Check Python terminal log.' });
    } finally {
        loadingSpinner.classList.add('hidden');
        analyzeButton.disabled = false;
    }
});

// --- Function to update the HTML with the analysis data ---
function displayResults(data) {
    const confidencePercent = (data.confidence * 100).toFixed(2) + '%';
    
    document.getElementById('disease-name').textContent = data.disease || 'Undetermined';
    document.getElementById('confidence').textContent = confidencePercent;
    
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
    
    apiDetailSection.classList.remove('hidden');

    if (data.disease !== 'Connection Failed') {
        showPage('remedy');
    }
    
    resultContent.classList.remove('hidden');
}