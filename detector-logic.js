// FINAL BACKEND URL (Render)
const BACKEND_API_URL = 'https://plant-disease-backend-2a4p.onrender.com/predict';

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

// -------------------- PAGE NAVIGATION --------------------

function showPage(pageId) {
    pageContents.forEach(page => page.classList.add('hidden'));
    const target = document.getElementById(`page-${pageId}`);
    if (target) target.classList.remove('hidden');

    navButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-page') === pageId) btn.classList.add('active');
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

// -------------------- IMAGE UPLOAD --------------------

fileInput.addEventListener('change', event => {
    uploadedFile = event.target.files[0];

    if (uploadedFile) {
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Leaf Image">`;
        };
        reader.readAsDataURL(uploadedFile);
        analyzeButton.disabled = false;
    } else {
        imagePreview.innerHTML = '<p>No image selected yet.</p>';
        analyzeButton.disabled = true;
    }

    resultsSection.classList.add('hidden');
    apiDetailSection.classList.add('hidden');
});

// -------------------- PREDICTION --------------------

analyzeButton.addEventListener('click', async () => {
    if (!uploadedFile) return;

    loadingSpinner.classList.remove('hidden');
    resultsSection.classList.remove('hidden');
    resultContent.classList.add('hidden');
    analyzeButton.disabled = true;

    const formData = new FormData();
    formData.append('image', uploadedFile);   // IMPORTANT: backend expects "image"

    try {
        const response = await fetch(BACKEND_API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) 
            throw new Error(`Server returned ${response.status}`);

        const data = await response.json();
        displayMask(data.mask);

    } catch (err) {
        console.error(err);
        alert("❌ Could not connect to backend!");
    } finally {
        loadingSpinner.classList.add('hidden');
        analyzeButton.disabled = false;
    }
});

// -------------------- DISPLAY MASK --------------------

function displayMask(maskArray) {
    const height = maskArray.length;
    const width = maskArray[0].length;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);

    // Fill canvas with mask pixels
    let index = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const v = maskArray[y][x]; // 0 or 255
            imgData.data[index++] = v;     // R
            imgData.data[index++] = v;     // G
            imgData.data[index++] = v;     // B
            imgData.data[index++] = 255;   // A
        }
    }

    ctx.putImageData(imgData, 0, 0);

    // Display on page
    document.getElementById('result-img').src = canvas.toDataURL();

    resultContent.classList.remove('hidden');
}
