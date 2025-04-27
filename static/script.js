async function predictEmotion() {
    const inputText = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');

    resultDiv.textContent = '';
    errorDiv.textContent = '';

    if (!inputText.trim()) {
        errorDiv.textContent = 'Please enter some text';
        return;
    }

    try {
        // Show loading text immediately
        resultDiv.textContent = 'Analyzing Sentiment...';

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        resultDiv.textContent = `Predicted Emotion: ${data.emotion}`;
    } catch (error) {
        resultDiv.textContent = '';  // clear "Analyzing..." if error
        errorDiv.textContent = 'Error predicting emotion. Please try again.';
        console.error('Error:', error);
    }
}
