
/**
 * @file dataService.js
 * @description Provides data management functions for the SIC/SOC Coding Tool.
 */



/**
 * Updated version of upsertRecord that accepts array of new records (backwards compatible)
 * @param {Array} data
 * @param {Array|Object} newRecords - array of new records. Will accept single record and convert to array
 * @param {function} getKey - a function determining name of id field
 * @returns {Array}
 */
 export function upsertRecords(data, newRecords, getKey = record => record?.id) {
    if (!Array.isArray(newRecords)) {
        newRecords = [newRecords];
    }

    const recordMap = new Map(
        data.filter(record => record && getKey(record) !== undefined)
            .map(record => [getKey(record), record])
    );

    newRecords.filter(record => record && getKey(record) !== undefined)
        .forEach(newRecord => {
            recordMap.set(getKey(newRecord), newRecord);
        });

    return [...recordMap.values()];
}

/**
 * Handles file selection and parses CSV data.
 * @param {Event} event - The file input change event.
 * @returns {Promise<Array>} A promise that resolves to an array of parsed job objects.
 */


export async function handleFixedWidthFileSelect(event) {

    const file = event.target.files[0];

    return new Promise((resolve, reject) => {
        //immeditely reject if theres no file or something wrong with the file
       if (!file) {
            reject(new Error("No file selected"));
        }

        //filereader to read in the file
        const reader = new FileReader();
        reader.onload = (event) => {
            //get the individual lines an discard any empty lines
            const lines = event.target.result.split('\n').filter(line => line.trim() !== '');
            //for each line do:
            const jobs = lines.map((row) => {

                //parse out the id
                //parse out the description and trim any excess whitespace
                const job = {
                    id: row.substring(0,7),
                    description: row.substring(7,49).trim(),
                    description_orig: row.substring(7,49).trim(),
                    code: '',
                    code_description: '',
                    code_score: '',
                    code_rank: ''
                }

                //store remaining string beyond character 49 in job['excess'] IF it exists
                if (row.length > 49){
                    if (row.substring(49).trim() !== '') {
                        job['excess'] = row.substring(49).trim()
                    }
                }
                return job;
            });
            //return the data
            resolve(jobs);
        };

        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

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
                        description: row[1].trim(),
                        description_orig: row[1].trim(),
                        code: '',
                        code_description: '',
                        code_score: '',
                        code_rank: ''
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
* Batches an array into chunks of a given size
* @param {Array} jobsData - The array of job objects
* @param {int} chunkSize - The size of each chunk
* @returns {Array} Array of arrays, each with length chunkSize
* NOTE:
* If the final chunk would be < chunkSize, the chunk will contain whatever items
* are left in the array
*/
function batchData(jobsData, chunkSize=20) {
    const chunkedData = [];
    for (let i = 0; i < jobsData.length; i += chunkSize) {
        chunkedData.push(jobsData.slice(i, i + chunkSize));
    };
    return chunkedData;
};

// Delay function to be used after receiving an error from the server
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


/**
 * Constructs a standardized mock response array for resultsData.
 * For each input object in the original request, creates a corresponding fake response
 * object with the original ID and a message.
 *
 * @param {Array} inputArray - Array of input objects, each containing an 'id' property
 * @param {object} [messageRecord] - Optional custom error message
 * @returns {Array} Array of error response objects, each containing:
 *                         - input_id: The original ID from the input object
 *                         - response: Array containing a single error message object
 */
export function constructMockResponse(inputArray,messageRecord={label: 'Loading...', description: 'Please wait',distance:0}) {
    return inputArray.map(item => ({
      input_id: item.id,
      response: [messageRecord]
    }));
  }

/**
 * Fetches code results from the server.
 * @param {Array} jobsData - The array of job objects.
 * @param {function} updateCallback - Function of (responseData,index) to run on each fetched result
 * @param {string} endpoint - The endpoint to query (default '/predict_soc')
 * @param {int} chunkSize - Size of batches
 * @param {int} retries - Number of retries to attempt
 *
 * First sorts the array in alphabetical order, then sets the localstorage to reflect.
 *
 * Then calls batchData to batch the jobsData array into chunks, then iterates
 * through each chunk and fetches the return, updating 'resultsData' with the result.
 *
 * Retry functionality is implemented in each call to the endpoint, trying up until 5
 * times to ensure a succesful response, otherwise continuing through to the next chunk
 */
export async function fetchResults(jobsData, updateCallback, sessionID, endpoint='/predict_soc',chunkSize=20,retries=5) {
    const chunkedData = batchData(jobsData, chunkSize);
    const failMessage = {label: 'Error', description: 'API call failed. Please click Re-search above',distance:0}
    chunkLoop: for (const [index, chunk] of chunkedData.entries()) {
        attemptLoop: for (let attempt = 1; attempt <=retries; attempt ++) {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(chunk),
            });
            if (response.status === 200) {
                let responseJson = await response.json();
                let responseData = responseJson?.data;
                updateCallback(responseData,index);
                console.log('Successfully processed chunk', index);
                postResultsData(sessionID, responseData)
                continue chunkLoop;
            } else {
                console.error(`Attempt ${attempt} of chunk ${index} failed:`, response.status);
                await delay(2000); // Waiting before continuing
                };
            };
            let failJson = constructMockResponse(chunk,failMessage);
            updateCallback(failJson,index);
            console.error('All attempts failed on chunk', index);
        };
    };

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
 * Download the data as a Fixed Width File
 * @param {Array} data
 * @param {string} filename
 */
export function downloadFixedWidthFile(data, filename = 'download.txt') {

    //set the fields we want to extract from the table object and asign a width for each one in the fieldWidths object
    const fieldsInOrder = ['id', 'description', 'description_orig', 'code', 'code_description', 'code_score','code_rank']
    const fieldWidths = {
        id: 7,
        description: 42,
        description_orig: 42,
        code: 5,
        code_description: 42,
        code_score: 5,
        code_rank: 5
    };

    //for each row in the data table process them to the string format and join the row strings with '\n'
    const txt = data.map(row => {

        //for each field/column in the row join them to one another with necessary padding
        const stringifiedRow = fieldsInOrder.map(field => {

            //get actual value at for that column/field and pad or truncate the string value to the asigned width for that
            const value = row[field]!=null ? String(row[field]) : "Null";
            const width = fieldWidths[field];
            return value.length > width ? value.substring(0,width) : value.padEnd(width, ' ');
        }).join('')

        return stringifiedRow
    }).join('\n')


    // Create a Blob containing the text data
    const blob = new Blob([txt], { type: 'text/csv;charset=utf-8;' });


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
        if (job.code && ignoreAssigned) return job;

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
                code: top.label.toString(),
                code_description: top.description,
                code_rank: top.rank.toString(),
                code_score: top.distance.toString()
            };
        }
        // Else don't update
        return job;
    });
}

/**
 * Posts results data to the specified endpoint for a given session.
 *
 * @param {string} sessionID - UUID identifying the current user session
 * @param {Array<Object>} resultsData - Array of result objects to be posted to the server
 * @param {string} [endpoint='/post_results'] - The API endpoint to post the results to
 * @returns {Promise<void>} - Promise that resolves when the post operation completes
 */
export async function postResultsData(sessionID, resultsData, endpoint='/post_results') {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([sessionID, resultsData]),
    });
    if (response.status === 200) {
        console.log("ResultsData posted successfully:", sessionID, resultsData)
    }
    else {
        console.log("ResultsData NOT posted successfully.")
    };
};

/**
 * Updates a single job, providing a jobID & given result to the specified endpoint for a given session.
 *
 * @param {string} sessionID - UUID identifying the current user session
 * @param {Object} updatedJobPayload - Object representing the payload from 'ASSIGN_RESULT', containing 'jobID' & 'result' as the fields
 * @param {string} [endpoint='/update_job'] - The API endpoint to update the job
 * @returns {Promise<void>} - Promise that resolves when the post operation completes
 */
export async function updateJobCode(sessionID, updatedJobPayload, endpoint='/update_job_code') {
    const response = await fetch(endpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([sessionID, updatedJobPayload]),
    });
    if (response.status === 200) {
        console.log("Job updated successfully:", updatedJobPayload)
    }
    else {
        console.log("Job NOT updated successfully.")
    };
};

/**
 * Sends session data and jobs information to the server via a POST request.
 *
 * @param {string} sessionID - The UUID that identifies the current session
 * @param {Array<Object>} jobsData - An array of objects containing job information to be posted
 * @param {string} [endpoint='/post_session'] - The server endpoint to post the data to
 * @returns {Promise<void>} - A promise that resolves when the POST request completes
 */
export async function postJobsData(sessionID, jobsData, endpoint='/post_session') {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([sessionID, jobsData]),
    });
    if (response.status === 200) {
        console.log("JOBSDATA POSTED SUCCESFULLY", sessionID, jobsData)
    }
    else {
        console.log("JOBSDATA NOT POSTED SUCCESFULLY")
    };
};


/**
 * Sends session data and jobs information to the server via a PUT request.
 *
 * @param {string} sessionID - The UUID that identifies the current session
 * @param {Array<Object>} jobsData - An array of objects containing job information to be updated
 * @param {string} [endpoint='/update_many_jobs'] - The server endpoint to post the data to
 * @returns {Promise<void>} - A promise that resolves when the POST request completes
 */
export async function updateJobsData(sessionID, jobsData, endpoint='/update_many_jobs') {
    const response = await fetch(endpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([sessionID, jobsData]),
    });
    if (response.status === 200) {
        console.log("JOBSDATA UPDATED SUCCESFULLY", sessionID, jobsData)
    }
    else {
        console.log("JOBSDATA NOT UPDATED SUCCESFULLY")
    };
};

/**
 * Requests Session Data: sessionID, jobsData, resultsData, from the server via GET request.
 *
 * @param {string} [endpoint='/get_state'] - The server endpoint to request the data from
 * @returns {Promise<void>} - A promise that resolves when the GET request completes. Containing 'sessionID', 'jobsData', 'resultsData'.
 */

export async function fetchSessionFromDatabase(endpoint='get_previous_session') {
    const response = await fetch(endpoint, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });
    if (response.status === 200) {
        const session_data = await response.json();
        return session_data
    } else {
        console.error("ERROR FETCHING PREVIOUS SESSION. (You may not be in a database supported environment).")
    };
};
