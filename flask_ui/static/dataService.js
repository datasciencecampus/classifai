/**
 * @file dataService.js
 * @description Provides data management functions for the SIC/SOC Coding Tool.
 */


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


/**
 * Autocodes jobs data using API results where confidence thresholds are met.
 * Only processes records that haven't already been coded (sic_code is empty).
 * Uses rank 1 and 2 responses to determine confidence.
 * @param {Array} jobsData - Array of job records with industry descriptions
 * @param {Array} resultsData - API response with ranked coding suggestions
 * @param {number} maxDistance - Maximum acceptable distance for rank 1 response
 * @param {number} minDiff - Required distance gap between rank 1 and 2 responses
 * @param {boolean} ignoreAssigned - If true, skip records with existing sic_code
 * @returns {Array} Updated jobsData with automatic codes applied where criteria met
 */
export function autocode(jobsData, resultsData, maxDistance=0.5, minDiff=0.05,ignoreAssigned=true) {
    // Create a map for quick lookups {"1":[responses],"2":[responses],...}
    const resultMap = Object.fromEntries(
        resultsData.map(r => [r.input_id, r.response])
    );

    // Loop through jobs data
    return jobsData.map(job => {
        // If already coded, don't update
        if (job.sic_code && ignoreAssigned) return job;

        const responses = resultMap[job.id];
        // If responses is zero length, don't update
        if (!responses?.length) return job;

        const top = responses.find(r => r.rank === 1);
        // If there is no rank-1 result or
        // the rank-1 result has distance > threshold, don't update
        if (!top || top.distance > maxDistance) return job;

        const second = responses.find(r => r.rank === 2);
        // If there is no rank-2 result or
        // the difference between rank-1 and rank-2 distance is greater than minDiff
        // update
        if (!second || (second.distance - top.distance) >= minDiff) {
            return {...job,
                sic_code: top.label.toString(),
                sic_code_description: top.description,
                sic_code_rank: top.rank.toString(),
                sic_code_score: top.distance.toString()
            };
        }
        // Else don't update
        return job;
    });
}
