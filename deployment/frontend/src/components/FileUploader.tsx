import { type ChangeEvent, useRef, useState } from 'react'

type PredictionResult = {
    label: string;
    confidence: number;
}

export default function FileUploader() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setResult(null);
        }
    }

    async function handleAnalyzeClick() {
        if (!file) return;

        setIsLoading(true);
        setResult(null);

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8001/predict', {
            method: 'POST',
            body: formData,
        });

        const data: PredictionResult = await response.json();
        setResult(data);
        setIsLoading(false);
    }

    function handleClearClick() {
        setFile(null);
        setResult(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }

    return (
        <div>
            <input type="file" onChange={handleFileChange} ref={fileInputRef} />
            {file && (
                <div>
                    <p>File name: {file.name}</p>
                    <p>Size: {(file.size / 1024).toFixed(2)} KB</p>
                    <p>Type: {file.type}</p>
                </div>
            )}
            <button onClick={handleAnalyzeClick} disabled={!file || isLoading}>
                {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>
            <button onClick={handleClearClick} disabled={!file}>
                Clear
            </button>
            {result && (
                <div style={{ color: result.label === 'Fake' ? 'red' : 'green' }}>
                    <p>Prediction: {result.label}</p>
                    <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                </div>
            )}
        </div>
    )
}