/**
 * @file dataService.js
 * @description Provides data management functions for the SIC/SOC Coding Tool.
 */

/**
 * Loads saved data from localStorage.
 * @returns {{jobs: Array, resultsData: Array, selectedJobId: string, selectedResult: Object}} The saved jobs and code results data.
 */
export function loadSavedData() {
    const jobs = JSON.parse(localStorage.getItem('jobsData')) || [];
    const resultsData = JSON.parse(localStorage.getItem('resultsData')) || [];
    const selectedJobId = JSON.parse(localStorage.getItem('selectedJobId')) || null;
    const selectedResult = JSON.parse(localStorage.getItem('selectedResult')) || {};
    return { jobs, resultsData, selectedJobId, selectedResult };
}

/**
 * Handles file selection and parses CSV data.
 * @param {Event} event - The file input change event.
 * @returns {Promise<Array>} A promise that resolves to an array of parsed job objects.
 */
export async function handleFileSelect(event) {
    const file = event.target.files[0];

    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: false,
            skipEmptyLines: true,
            error: (error, file) => reject(error),
            complete: (results) => {
                const jobs = results.data.map((row, index) => {
                    const job = {
                        id: row[0].trim(),
                        industry_description: row[1].trim(),
                        industry_description_orig: row[1].trim(),
                        sic_code: '',
                        sic_code_description: '',
                        sic_code_score: '',
                        sic_code_rank: ''
                    };
                    for (let i = 2; i < row.length; i++) {
                        job[`col${i-1}`] = row[i].trim();
                    }
                    return job;
                });
                resolve(jobs);
            }
        });
    });
}

/**
 * Fetches code results from the server.
 * @param {Array} jobsData - The array of job objects.
 * @returns {Promise<Object>} A promise that resolves to the fetched code results data.
 */
export async function fetchResults(jobsData) {
    try {
        const response = await fetch('/predict_sic', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobsData),
        });
        const responseJson = await response.json();
        localStorage.setItem('resultsData', JSON.stringify(responseJson.data));
        return responseJson.data;
    } catch (error) {
        console.error('Error:', error);
        return {};
    }
}

/**
 * Gets SIC/SOC results for a specific job.
 * @param {number} jobId - The ID of the job to get results for.
 * @param {Array} resultsData - the array of results
 * @returns {Array<Object>|null} The SIC/SOC results for the job, or null if not found.
 */
export function getResultsForJob(jobId,resultsData) {
    try {
        const results = resultsData.find(item => item.input_id === jobId);
        if (results && results.response) {
            return results.response.map(row => ({
                ...row,
                distance: parseFloat(row.distance).toFixed(2)
            }));
        }
    } catch (error) {
        console.error('Error in getResultsForJob:', error);
    }

    return null;
}

/**
 * Download the data as a CSV
 * @param {Array} data
 * @param {string} filename
 */
export function downloadCSV(data, filename = 'download.csv') {
    // Convert JSON to CSV using Papa Parse
    const csv = Papa.unparse(data,{
        header: true,
        delimiter: ",",
    });

    // Create a Blob containing the CSV data
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });

    // Create download link
    const link = document.createElement('a');

    // Create the URL for our blob
    const url = URL.createObjectURL(blob);

    // Set link properties
    link.setAttribute('href', url);
    link.setAttribute('download', filename);

    // Append link to body (required for Firefox)
    document.body.appendChild(link);

    // Trigger download
    link.click();

    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}
