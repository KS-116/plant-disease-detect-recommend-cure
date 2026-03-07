document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const previewContainer = document.getElementById('imagePreview');
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            
            previewContainer.innerHTML = `<img src="${e.target.result}" id="previewImg" style="max-width: 100%; max-height: 300px; border-radius: 10px; display: block;">`;
            
            previewContainer.style.display = "flex";
            previewContainer.style.justifyContent = "center";
            previewContainer.style.alignItems = "center";
            previewContainer.style.minHeight = "300px"; 
        }
        reader.readAsDataURL(file);
    }
});